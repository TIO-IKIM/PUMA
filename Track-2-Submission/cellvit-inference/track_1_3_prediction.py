import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import argparse
import json
import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from PIL import Image
import pandas as pd
import uuid
import ujson as json

from torch.utils.data import DataLoader, Dataset

from utils.postprocessing import DetectionCellPostProcessorCupy
from models.cell_segmentation.cellvit import CellViT
from models.cell_segmentation.cellvit_256 import CellViT256
from models.cell_segmentation.cellvit_sam import CellViTSAM
from models.classifier.linear_classifier import LinearClassifier
from utils.logger import Logger
from utils.tools import unflatten_dict
from config.templates import get_template_segmentation




class CellViTClassifierInferenceExperiment():
    def __init__(
        self,
        cellvit_path: Union[Path, str],
        classifier_path: Union[Path, str],
        gpu: int = 0
    ) -> None:
        self.logger: Logger
        self.model: nn.Module
        self.cellvit_model: nn.Module
        self.cellvit_run_conf: dict

        self.inference_transforms: Callable
        self.mixed_precision: bool
        self.num_classes: int

        self.classifier_path: Path
        self.cellvit_path: Path
        self.device: str

        self.cellvit_path = Path(cellvit_path)
        self.classifier_path = Path(classifier_path)
        self.device = f"cuda:{gpu}"

        self._instantiate_logger()
        self.cellvit_model, self.cellvit_run_conf = self._load_cellvit_model(
            checkpoint_path=self.cellvit_path
        )
        self.classifier_model, self.run_conf = self._load_model(checkpoint_path=self.classifier_path)
        self.num_classes = self.run_conf["data"]["num_classes"]
        self.inference_transforms = self._load_inference_transforms(
            normalize_settings_default=self.cellvit_run_conf["transformations"][
                "normalize"
            ],
            transform_settings=self.run_conf.get("transformations", None),
        )
        self._setup_amp(enforce_mixed_precision=False)

    def _instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            comment="inference",
        )
        self.logger = logger.create_logger()

    def _load_model(self, checkpoint_path: Union[Path, str]) -> Tuple[nn.Module, dict]:
        """Load the Classifier Model

        checkpoint_path (Union[Path, str]): Path to a checkpoint

        Returns:
            Tuple[nn.Module, dict]:
                * Classifier
                * Configuration for training the classifier
        """
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        run_conf = unflatten_dict(model_checkpoint["config"], ".")

        model = LinearClassifier(
            embed_dim=model_checkpoint["model_state_dict"]["fc1.weight"].shape[1],
            hidden_dim=run_conf["model"].get("hidden_dim", 100),
            num_classes=run_conf["data"]["num_classes"],
            drop_rate=0,
        )
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        model = model.to(self.device)
        model.eval()
        return model, run_conf

    def _load_cellvit_model(
        self, checkpoint_path: Union[Path, str]
    ) -> Tuple[nn.Module, dict]:
        """Load a pretrained CellViT model

        Args:
            checkpoint_path (Union[Path, str]): Path to a checkpoint

        Returns:
            Tuple[nn.Module, dict]:
                * CellViT-Model
                * Dictionary with CellViT-Model configuration
        """
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # unpack checkpoint
        cellvit_run_conf = unflatten_dict(model_checkpoint["config"], ".")
        model = self._get_cellvit_architecture(
            model_type=model_checkpoint["arch"], model_conf=cellvit_run_conf
        )
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        cellvit_run_conf["model"]["token_patch_size"] = model.patch_size
        model = model.to(self.device)
        model.eval()
        return model, cellvit_run_conf

    def _get_cellvit_architecture(
        self,
        model_type: Literal[
            "CellViT",
            "CellViT256",
            "CellViTSAM",
        ],
        model_conf: dict,
    ) -> Union[
        CellViT, CellViT256, CellViTSAM
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2

        Returns:
            Union[CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViT256",
            "CellViTSAM",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT"]:
            model = CellViT(
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                embed_dim=model_conf["model"]["embed_dim"],
                input_channels=model_conf["model"].get("input_channels", 3),
                depth=model_conf["model"]["depth"],
                num_heads=model_conf["model"]["num_heads"],
                extract_layers=model_conf["model"]["extract_layers"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256"]:
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                vit_structure=model_conf["model"]["backbone"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )
        
        return model

    def _load_inference_transforms(
        self,
        normalize_settings_default: dict,
        transform_settings: dict = None,
    ) -> Callable:
        """Load inference transformations

        Args:
            normalize_settings_default (dict): Setting of cellvit model
            transform_settings (dict, optional): Alternative to overwrite. Defaults to None.

        Returns:
            Callable: Transformations
        """
        self.logger.info("Loading inference transformations")

        if transform_settings is not None and "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = normalize_settings_default["mean"]
            std = normalize_settings_default["std"]
        inference_transform = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])
        return inference_transform

    def _setup_amp(self, enforce_mixed_precision: bool = False) -> None:
        """Setup automated mixed precision (amp) for inference.

        Args:
            enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        if enforce_mixed_precision:
            self.mixed_precision = enforce_mixed_precision
        else:
            self.mixed_precision = self.run_conf["training"].get(
                "mixed_precision", False
            )

    def _get_cellvit_result(
        self,
        image: torch.Tensor,
        postprocessor: DetectionCellPostProcessorCupy,
    ) -> Tuple[
        List[dict], List[dict], dict[dict], List[float], List[float], List[float]
    ]:
        # TODO docstring and return types
        # return lists
        self.logger.info("Perform CellViT prediction ...")
        overall_extracted_cells = []
        image_pred_dict = {}

        image = image.to(self.device)[None, :]
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = self.cellvit_model.forward(image, retrieve_tokens=True)
        else:
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        # transform predictions and create tokens
        predictions = self._apply_softmax_reorder(predictions)
        _, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self._extract_tokens(cell_pred_dict, predictions, list(image.shape[2:]))


        return (
            cell_pred_dict[0],
            tokens[0]
        )

    def _apply_softmax_reorder(self, predictions: dict) -> dict:
        """Reorder and apply softmax on predictions

        Args:
            predictions(dict): Predictions

        Returns:
            dict: Predictions
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        return predictions


    def _extract_tokens(
        self, cell_pred_dict: dict, predictions: dict, image_size: List[int]
    ) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict
            image_size (List[int]): Image size of the input image (H, W)

        Returns:
            List: List of topkens for each patch
        """
        if hasattr(self.cellvit_model, "patch_size"):
            patch_size = self.cellvit_model.patch_size
        else:
            patch_size = 16

        if patch_size == 16:
            rescaling_factor = 1
        else:
            if image_size[0] == image_size[1]:
                if image_size[0] in self.cellvit_model.input_rescale_dict:
                    rescaling_factor = (
                        self.cellvit_model.input_rescale_dict[image_size] / image_size
                    )
                else:
                    self.logger.error(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
                    raise RuntimeError(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
            else:
                self.logger.error(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )
                raise RuntimeError(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )

        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = rescaling_factor * cell["bbox"]
                bb_index = bbox / patch_size
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[
                    :, bb_index[0, 0] : bb_index[1, 0], bb_index[0, 1] : bb_index[1, 1]
                ]
                cell_token = torch.mean(
                    rearrange(cell_token, "D H W -> (H W) D"), dim=0
                )
                extracted_cell_tokens.append(cell_token.detach().cpu())
            batch_tokens.append(extracted_cell_tokens)

        return batch_tokens

    def _get_classifier_result(
        self, cell_tokens: list[torch.Tensor]
    ) -> dict:
        """Get classification results for extracted cells

        Args:
            cell_tokens (List[dict]): List of extracted cell tokens, one token for each cell in list
        Returns:
        """
        self.logger.info("Perform classifier cell class prediction ...")
        cell_tokens = torch.stack(cell_tokens, axis=0)
        cell_tokens = cell_tokens.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                logits = self.classifier_model.forward(cell_tokens)
        else:
            # make predictions
            logits = self.model.forward(cell_tokens)
        probs = F.softmax(logits, dim=1)
        class_predictions = torch.argmax(probs, dim=1)

        return class_predictions


    def _create_polygon_json(self, cell_dict):
        # Define colors and class names for the different classes
        colors = {
            1: (0, 255, 0),  # lymphocytes
            2: (255, 0, 0),  # tumor
            3: (0, 0, 255)   # other
        }
        class_names = {
            1: 'nuclei_lymphocyte',
            2: 'nuclei_tumor',
            3: 'nuclei_other'
        }

        output_json = {
            "type": "Multiple polygons",
            "polygons": []
        }
        for cell_id, cell in cell_dict.items():
            path_points = [[float(pt[0]), float(pt[1]), 0.5] for pt in cell["contour"]]
            cell_type = cell["type"]

            # create a polygon entry
            polygon_entry = {
                "name": class_names[cell_type],
                "seed_point": path_points[0],  # using first point as the seed point
                "path_points": path_points,
                "sub_type": "",  # empty string for subtype
                "groups": [],  # empty array for groups
                "probability": 1  # confidence score
            }
            output_json["polygons"].append(polygon_entry)

        output_json["version"] = {
            "major": 1,
            "minor": 0
        }

        return output_json


    def _convert_json_geojson(
        self, cell_list: list[dict]
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for segmentation:
                    * type
                    * contour
        Returns:
            List[dict]: Geojson like list
        """
        color_dict = {
            1: (0, 255, 0),  # lymphocytes
            2: (255, 0, 0),  # tumor
            3: (0, 0, 255)   # other
        }
        class_names = {
            1: 'nuclei_lymphocyte',
            2: 'nuclei_tumor',
            3: 'nuclei_other'
        }

        cell_segmentation_df = pd.DataFrame(cell_list).T
        detected_types = sorted(cell_segmentation_df.type.unique())
        geojson_placeholder = []
        for cell_type in detected_types:
            cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
            contours = cells["contour"].to_list()
            final_c = []
            for c in contours:
                c = [[int(x), int(y)] for x,y in list(c)]
                c.append(c[0])
                final_c.append([c])
            cell_geojson_object = get_template_segmentation()
            cell_geojson_object["id"] = str(uuid.uuid4())
            cell_geojson_object["geometry"]["coordinates"] = final_c
            cell_geojson_object["properties"]["classification"][
                "name"
            ] = class_names[cell_type]
            cell_geojson_object["properties"]["classification"][
                "color"
            ] = color_dict[cell_type]
            geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder

    def run_inference(self, input_file, output_file):
        """Run Inference on Test Dataset"""
        extracted_cells = []  # all cells detected with cellvit
        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)

        input_image = np.array(Image.open(input_file).convert("RGB"))
        input_image = self.inference_transforms(image=input_image)["image"]
        # Step 1: Extract cells with CellViT
        with torch.no_grad():
            (
                cell_pred_dict,
                cell_tokens
            ) = self._get_cellvit_result(
                image=input_image,
                postprocessor=postprocessor,
            )
            class_predictions = self._get_classifier_result(cell_tokens=cell_tokens)

        # update cell dict
        final_cell_dict = {}
        for cell_idx, cell_props in cell_pred_dict.items():
            cell_props["type"] = int(class_predictions[cell_idx-1]) + 1 # add +1 because of naming convention
            final_cell_dict[cell_idx] = cell_props

        # create polygon json
        polygon_json = self._create_polygon_json(cell_dict=final_cell_dict)
        with open(output_file, "w") as f:
            json.dump(polygon_json, f, indent=2)

        # visualize (.geojson)
        # geojson_string = self._convert_json_geojson(final_cell_dict)
        # with open("test.geojson", "w") as f:
        #     json.dump(geojson_string, f, indent=2)
        self.logger.info("Finished")

class CellViTInfExpSegmentationParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for PUMA",
        )
        parser.add_argument(
            "--cellvit_path", type=str, help="Path to the Cellvit model"
        )
        parser.add_argument(
            "--classifier_path", type=str, help="Path to the classifier model"
        )
        parser.add_argument(
            "--input_file", type=str, help="Path to input file (.png or .tiff)"
        )
        parser.add_argument(
            "--output_file", type=str, help="Path to output file .json"
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpSegmentationParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTClassifierInferenceExperiment(
        cellvit_path=configuration["cellvit_path"],
        classifier_path=configuration["classifier_path"],
    )
    experiment_inferer.run_inference(
        input_file=configuration["input_file"],
        output_file=configuration["output_file"]
    )
