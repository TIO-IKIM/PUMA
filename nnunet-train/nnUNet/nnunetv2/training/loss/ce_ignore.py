import torch
from torch import nn, Tensor
import numpy as np

class CrossEntropyLossIgnoreBase(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(ignore_index=12)
    def forward(self, input, target):
        # Squeeze the input tensor along the specified dimension
        input_tensor = input[0]
        target_tensor = target[0].squeeze()
        # Call the superclass forward function with modified input
        return super().forward(input_tensor, target_tensor.long())