import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.ce_ignore import CrossEntropyLossIgnoreBase

class nnUNetTrainerIgnoreIndexBase(nnUNetTrainer):
    def _build_loss(self):
        loss = CrossEntropyLossIgnoreBase()

        return loss