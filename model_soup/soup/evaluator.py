import torch


class Evaluator:
    def __init__(self, device):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def eval_func(self, model):
        raise NotImplemented