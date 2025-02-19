#!/usr/bin/env bash

# get the image file (uuid.tiff?)
input_file=$(find /input/images/melanoma-wsi -type f | head -n 1)

# Check if a file was found
if [ -z "$input_file" ]; then
  echo "No file found in /input/images/melanoma-wsi"
  exit 1
fi

img_name=$(basename "$input_file")
mkdir -p /output/images/melanoma-tissue-mask-segmentation

# Step 1: perform tissue inference
echo "Processing nnUNet"
python ./nnunet-inference/inference/inference_singlemodel.py \
    --plan_path  ./nnunet-inference/weights/Dataset998_PUMA/nnUNetTrainer__nnUNetPlans__2d_512/ \
    --input_file "$input_file" \
    --output_file /output/images/melanoma-tissue-mask-segmentation/${img_name}

# Step 2: perform nuclei inference
python ./cellvit-inference/track_2_10_prediction.py \
    --cellvit_path ./cellvit-inference/checkpoint/CellViT-SAM-H-x40-AMP.pth\
    --classifier_path ./cellvit-inference/checkpoint/classifier-track2-10-classes.pth \
    --input_file "$input_file" \
    --output_file /output/melanoma-10-class-nuclei-segmentation.json

echo "Done"