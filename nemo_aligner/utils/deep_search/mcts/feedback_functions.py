import re

import pandas as pd
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer, math_equal


class Feedback(object):
    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError


class GSK8KFeedbackDataset(Feedback):
    def score(self, response, answer):
        """
        score the response
        """
        # this needs to be on a seperate server for anything
        # complicated but for GSM8K this is fine
        response = extract_answer(response)
        return float(math_equal(response, answer))


class GSK8KFeedback(Feedback):
    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedback(Feedback):
    def __init__(self):
        ...

    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedbackHF(Feedback):
    def __init__(self, split):
        super().__init__()
        self.ds = load_dataset("gsm8k", "main")
        self.split = split

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.ds[self.split][data_id]["answer"].lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0
