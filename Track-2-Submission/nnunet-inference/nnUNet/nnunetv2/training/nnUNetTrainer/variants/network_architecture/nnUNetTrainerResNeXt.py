from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn
import torch
import segmentation_models_pytorch as smp
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerResNeXt(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if 'norm_op' not in arch_init_kwargs.keys():
            raise RuntimeError("'norm_op' not found in arch_init_kwargs. This does not look like an architecture "
                               "I can hack BN into. This trainer only works with default nnU-Net architectures.")

        from pydoc import locate
        conv_op = locate(arch_init_kwargs['conv_op'])
        bn_class = get_matching_batchnorm(conv_op)
        arch_init_kwargs['norm_op'] = bn_class.__module__ + '.' + bn_class.__name__
        arch_init_kwargs['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}

        model = smp.UnetPlusPlus(
            encoder_name="timm-res2net50_26w_4s",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=num_input_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_output_channels,                      # model output channels (number of classes in your dataset)
        )

        return model

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss = RobustCrossEntropyLoss(
            weight=torch.tensor([25.64, 1, 2.88, 27.37, 109.47, 68.28]).to(device), ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss