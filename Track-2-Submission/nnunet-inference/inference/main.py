import os
from inference_singlemodel import tissue_segmentation_singlemodel
from inference_ensemble import tissue_segmentation_ensemblemodel
import logging

def main(base_dir: str, in_path: str, out_path: str, model_name: str, dataset_name: str, model_name2: str=None) -> None:
    """Performs tissue segmentation on input images using a single model or an ensemble of models.

    Args:
        in_path (str): Relative path to the input directory containing image files.
        out_path (str): Relative path to the output directory for saving segmented images.
        model_name (str): Name of the primary model for segmentation.
        dataset_name (str): Name of the dataset directory containing model weights.
        model_name2 (str, optional): Name of the secondary model for ensemble segmentation. Defaults to None.

    Returns:
        None
    """
    # Construct absolute paths
    in_path = os.path.join(base_dir, "PUMA/Tissue/nnUnet_raw", in_path)
    out_path = os.path.join(base_dir, "puma/outputs", out_path)
    model_path = os.path.join(base_dir, "puma/weights", dataset_name, model_name)

    # Ensure output directory exists
    os.makedirs(out_path, exist_ok=True)

    # List files in input directory
    filenames = os.listdir(in_path)

    for filename in filenames:
        ct_path = os.path.join(in_path, filename)
        seg_output = os.path.join(out_path, os.path.splitext(filename)[0] + ".tif")

        # Perform segmentation
        if model_name2:
            tissue_segmentation_ensemblemodel(
                model_path1=model_name, 
                model_path2=model_name2, 
                file_in=ct_path, 
                file_out=seg_output
            )
        else:
            tissue_segmentation_singlemodel(
                model_path=model_path, 
                file_in=ct_path, 
                file_out=seg_output
            )

if __name__ == "__main__":
    main(
        base_dir="/opt/app/nnunet-inference",
        in_path="Dataset998_PUMA/imagesTs", 
        out_path="weightedceloss", 
        model_name="nnUNetTrainerWeightedCELoss__nnUNetPlans__2d", 
        dataset_name="Dataset998_PUMA", 
        model_name2=None  # Only for ensemble
    )
