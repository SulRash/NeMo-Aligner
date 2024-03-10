# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import numpy as np
import torch
import torch.nn.functional as F

from nemo_aligner.utils.deep_search.mcts.mcts import Node


class SearchDB:
    def __init__(self):
        self.db = {}
        # node_id
        self.node_ids = {}
        self.inference_params = {}
        self.attention_mask = {}
        self.position_ids = {}

    def add_init_obj(self, session_id, inference_params, attention_mask, position_ids):
        self.inference_params[session_id] = inference_params
        self.attention_mask[session_id] = attention_mask
        self.position_ids[session_id] = position_ids

    def add_root(self, session_info, context_id, node):
        if session_info not in self.db:
            session = {}
            self.db[session_info] = session
        else:
            session = self.db[session_info]
        for token in context_id[:-1]:
            if token not in session:
                session[token] = Node(None, None, None, None, None, None)
            session = session[token].children
        session[context_id[-1]] = node

    def get(self, session_info, context_id):
        if session_info not in self.db:
            raise ValueError(f"{session_info} not in db")
        db = self.db[session_info]
        for token in context_id[:-1]:
            db = db[token].children
        return db[context_id[-1]]

    def get_infer_cache(self, session_info, context_id, action):
        if session_info not in self.db:
            return None
        db = self.db[session_info]
        for token in context_id:
            if token not in db:
                return None
            db = db[token].children
        if action not in db:
            return None
        node = db[action]
        if node.value_sum is None:
            return None

        output = {}
        output["value"] = node.value_sum
        policy = node.prior[0]
        actions = node.prior[1]
        # actions = []
        # policy = []
        # for child_action in node.children:
        #     child = node.children[child_action]
        #     assert child.action == child_action
        #     actions.append(child.action)
        #     policy.append(child.prior)
        output["action"] = actions
        output["policy"] = policy
        output["value"] = np.array(node.value_sum)
        return output

    def get_attention_mask(self, session_id):
        return self.attention_mask[session_id]

    def get_position_ids(self, session_id):
        return self.position_ids[session_id]

    def get_inference_params(self, session_id):
        return self.inference_params[session_id]

    def add_inference_params(self, session_id, inference_params):
        self.inference_params[session_id] = inference_params

    def delete(self, session_id):
        del self.db[session_id]
        self.node_id = 0
        del self.inference_params[session_id]
        del self.attention_mask[session_id]
        del self.position_ids[session_id]


def _get_trailing_padding(tokens):
    counts = 0
    for token in tokens[::-1]:
        if token == 0:
            counts += 1
        else:
            break
    return counts


def get_kv_cache(selected_actions, session_info, context_ids, search_db: SearchDB, pad_id: int = 0):
    batched_kv_cache = []
    batched_tokens = []
    for action, context_id in zip(selected_actions, context_ids):
        action = action.tolist()
        # reverse the order
        action.reverse()
        node = search_db.get(session_info, context_id)
        # assert node.action == action
        tokens = []
        tokens.extend(action)
        new_kv_cache = {key: [] for key in node.state.keys()}
        #         while node.parent is not None:
        while True:
            # if node.action is List
            if isinstance(node.action, list):
                # make a copy
                context_tokens = node.action.copy()
                # reverse the order
                context_tokens.reverse()
                tokens.extend(context_tokens)
            else:
                tokens.append(node.action)
            for key in node.state.keys():
                new_kv_cache[key].append(node.state[key])
            if node.parent is None:
                break
            node = node.parent
        # reverse the tokens order
        tokens.reverse()
        for key in new_kv_cache.keys():
            list_kv = new_kv_cache[key]
            keys = [item[0] for item in list_kv]
            vals = [item[1] for item in list_kv]
            # reverse the order
            keys.reverse()
            vals.reverse()
            keys = torch.concatenate(keys, axis=0)
            vals = torch.concatenate(vals, axis=0)
            new_kv_cache[key] = (keys, vals)
        batched_kv_cache.append(new_kv_cache)
        batched_tokens.append(np.array(tokens))
    # get number of trailing padding
    trailing_padding = [_get_trailing_padding(tokens) for tokens in batched_tokens]
    tokens_nums = [len(tokens) for tokens in batched_tokens]
    no_padding_length = [i - j for i, j in zip(tokens_nums, trailing_padding)]

    paddings = max(no_padding_length) - np.array(no_padding_length)
    token_list = []
    for token, padding, no_padding in zip(batched_tokens, paddings, no_padding_length):
        padding = padding.item()
        token = np.pad(token[0:no_padding], ((0, padding),), "constant", constant_values=pad_id)
        token_list.append(token)
    # all the tokens should have the same length, including context and padding
    tokens = np.stack(token_list, axis=0)
    # find the maximum length
    max_length = tokens.shape[1]

    # the context length
    context_length = np.array([len(c) for c in context_ids])
    # find out the minimum context length within the batch
    min_depth = context_length.min()
    # the minimum tokens that need to be passed in for inference
    tokens = tokens[:, min_depth:]
    # kv cache padding
    paddings = max_length - context_length
    # concat kv cache
    output_kv_cache = {}
    for key in batched_kv_cache[0].keys():
        keys = []
        vals = []
        for item, padding in zip(batched_kv_cache, paddings):
            padding = padding.item()
            key_item = item[key][0][:, None, ...]
            key_item = F.pad(key_item, (0, 0, 0, 0, 0, 0, 0, padding + 1), "constant", 0)
            keys.append(key_item)
            value_item = item[key][1][:, None, ...]
            value_item = F.pad(value_item, (0, 0, 0, 0, 0, 0, 0, padding + 1), "constant", 0)
            vals.append(value_item)
        keys = torch.concatenate(keys, axis=1)
        vals = torch.concatenate(vals, axis=1)
        output_kv_cache[key] = (keys, vals)
    return output_kv_cache, tokens
