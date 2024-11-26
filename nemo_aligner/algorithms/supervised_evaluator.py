# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from statistics import mean

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.utils import logging
from nemo_aligner.metrics import InferenceMetricsHandler
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.trainer_utils import compute_limit_batches, compute_num_steps_per_epoch

IMAGE_CAPTION_KEY = "images_and_captions"


class SupervisedTrainer:
    """trainer that implements the supervised training loop
        this is useful for things like SFT and reward model training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.step = 0
        self.consumed_samples = 0

        self.ckpt_callback = ckpt_callback

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(self.train_dataloader.batch_sampler)

        self.limit_val_batches = compute_limit_batches(len(val_dataloader), self.cfg.limit_val_batches)
        self.set_max_steps()

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

        # any metrics that require running full token-by-token inference during validation
        self.inference_metrics_handler = InferenceMetricsHandler(cfg.get("inference_metrics"))

    def validation_step(self, batch):
        self.model.prepare_for_validation_step()

        loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=True)

        self.model.finish_validation_step()
        return loss_mean, metrics

    @torch.no_grad()
    def run_validation(self):
        loss_means = []
        val_metrics = defaultdict(list)

        val_pbar = tqdm(
            zip(range(self.limit_val_batches), self.val_dataloader),
            total=self.limit_val_batches,
            leave=True,
            desc="Validation steps",
        )

        for _, batch in val_pbar:
            self.timer.start("validation_step_time")
            loss_mean, metrics = self.validation_step(batch)
            self.timer.stop("validation_step_time")
            validation_step_time = self.timer.get("validation_step_time")
            metrics["validation_step_time"] = validation_step_time

            if self.inference_metrics_handler.has_metrics():
                generation_output = self.run_generation(batch)
                self.inference_metrics_handler.update(batch, generation_output)

            # for stable diffusion logging
            if IMAGE_CAPTION_KEY in metrics:
                images, captions = metrics.pop(IMAGE_CAPTION_KEY)
                self.logger.log_image(key="validation images", images=images, caption=captions)

            loss_means.append(loss_mean)
            for k, v in metrics.items():
                val_metrics[k].append(v)
            log_val_metrics = {f"val_{k}": v for k, v in metrics.items()}
            val_pbar.set_postfix(log_val_metrics)

        val_metrics = {k: mean(v) for k, v in val_metrics.items()}
        val_metrics.update(self.inference_metrics_handler.compute())
        self.inference_metrics_handler.reset()

        return mean(loss_means), val_metrics

    @torch.no_grad()
    def run_generation(self, batch):
        return self.model.infer({"text": batch["contexts"], "length": batch["context_lengths"]})

    def fit(self):
        logging.info("Running validation in the very beginning.")
        val_loss, val_metrics = self.run_validation()
        # validation is done on the UPDATED weights
        # so we use the incremented self.step
        self.logger.log_metrics(val_metrics, step=self.step, prefix="val/")
        logging.info("Initial validation metrics logged.")

        self.logger.finalize()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]

        loaded_values = [self.step, self.consumed_samples]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    @property
    def epoch(self):
        return self.step // self.num_steps_per_epoch
