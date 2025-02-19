import os
import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import re

def predict_nnunet(model_training_output_dir, img, props):

    folds = ()
    file_list = os.listdir(model_training_output_dir)
    for file in file_list:
        match = re.search(r'fold_(\d+)', file)
        if match:
            folds += (int(match.group(1)),)


    checkpoint_name = 'checkpoint_best.pth' 

    # Create predictor instance
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0), 
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    # Initialize from trained model folder
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir,
        use_folds=folds,
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
 

