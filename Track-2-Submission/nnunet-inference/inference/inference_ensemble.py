# from env_setup import setup_nnunet
# setup_nnunet()

import argparse
import numpy as np
import os
from PIL import Image
import tifffile

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import re

def predict_nnunet(plan_path, img, props):
    checkpoint_name = 'checkpoint_best.pth' 

    # Create predictor instance
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        #perform_everything_on_device=True,
        device=torch.device('cuda', 0), 
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=False
    )

    # Initialize from trained model folder
    predictor.initialize_from_trained_model_folder(
        plan_path,
        use_folds=[0,1,2,3,4],
        checkpoint_name=checkpoint_name
    )

    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (3, 0,  1, 2)) 

    # Perform prediction
    prediction = predictor.predict_single_npy_array(
        input_image=img,
        image_properties=props,
        output_file_truncated=None,
        save_or_return_probabilities=False
    )

    return prediction
 




def puma_challenge_label(label: np.ndarray) -> np.ndarray:

    label[label == 1] = 30
    label[label == 2] = 10
    label[label == 3] = 40
    label[label == 4] = 20
    
    label[label == 10] = 1
    label[label == 20] = 2
    label[label == 30] = 3
    label[label == 40] = 4

    return label

def tissue_segmentation_singlemodel(file_in: str, file_out: str, plan_path: str) -> np.ndarray:
    """Performs tissue segmentation on a single image using a specified model.

    Args:
        file_in (str): Path to the input image file.
        file_out (str): Path to save the segmented output image.
        model_path (str): Path to the directory containing the trained model weights.

    Returns:
        np.ndarray: Segmented label array for the input image.
    """
    # Load the input image
    img = Image.open(file_in).convert("RGB")

    # Define image properties for the prediction
    props = {
        'spacing': [1, 1, 1],  # Assuming isotropic spacing
        'size': [img.size]     # Image dimensions
    }

    # Perform model inference
    label = predict_nnunet(plan_path, img, props)

    label = puma_challenge_label(label).astype(np.uint8).squeeze()
    print(label.shape)

    with tifffile.TiffWriter(file_out) as tif:
        tif.write(
            label,
            resolution=(300, 300),
            extratags=[
                ('MinSampleValue', 'I', 1, int(1)),
                ('MaxSampleValue', 'I', 1, int(label.max()))
            ]
        )



class NNUnetParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Parse nnUNet",
        )
        parser.add_argument(
            "--plan_path", type=str, help="Path to the nnunet model"
        )
        parser.add_argument(
            "--input_file", type=str, help="Path to input file (.tiff)"
        )
        parser.add_argument(
            "--output_file", type=str, help="Path to output file (.tiff)"
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = NNUnetParser()
    configuration = configuration_parser.parse_arguments()
    tissue_segmentation_singlemodel(
        file_in=configuration["input_file"],
        file_out=configuration["output_file"],
        plan_path=configuration["plan_path"]
    )