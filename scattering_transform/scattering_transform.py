import torch
from torch import nn
import torch.nn.functional as F

class ScatteringTransform(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, questions, answers):
        return questions
