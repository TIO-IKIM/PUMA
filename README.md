[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2501.05269)

# PUMA Challenge solution IKIM

> [!IMPORTANT]
> A full version of the code including model checkpoints is hosted on Zenodo

This repository contains the training instructions as well as the Dockerized environment for running the PUMA challenge evaluation for track 1 and track 2 for the team "UME". The container includes all necessary dependencies to execute the model and run inference on input data. The training instructions are provided in the train folder. 

## Team Members

All participants are affiliated with the Institute for Artificial Intelligence in Medicine at the University Hospital Essen

- Negar Shamiri
- Moritz Rempe
- Lukas Heine
- Fabian Hörst
- Jens Kleesiek (Prof.)

## Acknowledgements

We thank the PUMA challenge organizers for providing the dataset and organizing this valuable competition. If you use this code or the PUMA dataset in your research, please cite the following work:


Schuiveling, M., Liu, H., Eek, D., Breimer, G. E., Suijkerbuijk, K. P. M., Blokx, W. A. M., & Veta, M. (2024). A Novel Dataset for Nuclei and Tissue Segmentation in Melanoma with baseline nuclei segmentation and tissue segmentation benchmarks. Cold Spring Harbor Laboratory. https://doi.org/10.1101/2024.10.07.24315039

```bibtex
@article{Schuiveling2024,
  title = {A Novel Dataset for Nuclei and Tissue Segmentation in Melanoma with baseline nuclei segmentation and tissue segmentation benchmarks},
  url = {http://dx.doi.org/10.1101/2024.10.07.24315039},
  DOI = {10.1101/2024.10.07.24315039},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Schuiveling,  Mark and Liu,  Hong and Eek,  Daniel and Breimer,  Gerben E. and Suijkerbuijk,  Karijn P.M. and Blokx,  Willeke A.M. and Veta,  Mitko},
  year = {2024},
  month = oct 
}
```


## Summary
Our solution consists of two frameworks (1) nnUNet for the tissue segmentation and (2) CellViT++ finetuned on the nuclei classes.The nnUNet was first pretrained using an public NSCLC tissue segmenation dataset and then finetuned on the PUMA data. CellViT++ uses the CellViT-SAM-H model as the backbone and we just trained a simple cell classifier on top as explained in the CellViT++ publication. If you have problems using either of the frameworks, we recommend to check out both repos:
- nnUNet: https://github.com/MIC-DKFZ/nnUNet
- CellViT++: https://github.com/TIO-IKIM/CellViT-plus-plus

All checkpoints can either be found in our [GDrive](https://drive.google.com/drive/folders/1enKbMiYK7gnbsL2nn06R146XTFo5Uq46) folder or on Zenodo.


## Docker creation for Inference (submission)

### Prerequisites

#### nnU-Net Checkpoints
Download the tissue segmentation checkpoint from our [GDrive](https://drive.google.com/drive/folders/1enKbMiYK7gnbsL2nn06R146XTFo5Uq46) and place it in:
- Track 1: `./Track-1-Submission/nnunet-inference/weights/Dataset998_PUMA/nnUNetTrainer__nnUNetPlans__2d_512/fold_0/checkpoint_best.pth`
- Track 2: `./Track-2-Submission/nnunet-inference/weights/Dataset998_PUMA/nnUNetTrainer__nnUNetPlans__2d_512/fold_0/checkpoint_best.pth`

#### CellViT++ Checkpoints
1. Download all CellViT++ checkpoints from the same Google Drive folder (inside `cellvit++` directory)
2. Place them in:
   - Track 1: `./Track-1-Submission/cellvit-inference/checkpoint/`
   - Track 2: `./Track-2-Submission/cellvit-inference/checkpoint/`



### Building the container

#### Track 1
```bash
cd Track-1-Submission
docker build -t select-your-name-track-1 .
docker save select-your-name-track-1 | gzip -c > select-your-name-track-1.tar.gz
```

#### Track 2
```bash
cd Track-2-Submission
docker build -t select-your-name-track-2 .
docker save select-your-name-track-2 | gzip -c > select-your-name-track-2.tar.gz
```

## Training 

### nnU-Net
> [!IMPORTANT]
> Please use the nnUNet-Version provided in the nnunet-train folder and install this version. We used a small adaption such that installing from source needs to be done. Follow the following instruction to install the nnUNet: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md (integrative framework)

Steps for the installation:
1. Install PyTorch
2. Install as integrative framework:
   1. `cd ./nnunet-train/nnUNet`
   2. `pip install -e .`
3. Set your environment variables

if you have no experience using nnUnet it is adivsed to follow their tutorial first as explaining the nnUNet here would exceed the scope of this repo.
See here: https://github.com/MIC-DKFZ/nnUNet/tree/master?tab=readme-ov-file#how-to-get-started

#### Training Procedure
The training code to reproduce the results is placed within the nnunet-train folder.

Steps to reproduce:
1. Install the nnUNet as explained above
2. Download the pretraining data (NSCLC) and prepare them
3. Download the PUMA data and preprare them
4. Pretrain nnUNet using the NSCLC data (one fold)
5. Finetune for PUMA (one fold)

##### 1. Download and prepare pre-training dataset
Please download the data from zenodo (https://zenodo.org/records/12818382). Transform the data into the nnUNet format. The dataset json is given in `./nnunet-train/pretrain/configurations`

The final folder should look similar to this:
```markdown
Dataset999_Lung
├── imagesTr
│   ├── identifier1_0000.png
│   └── ...
├── labelsTr
│   ├── identifier1.png
│   └── ...
└── dataset.json
```

##### 2. Download and prepare fine-tuning dataset (PUMA)
Download the dataset from the challenge organizers and bring it into the nnunet format. To do so, use the provided script by the challenge organizers to convert .geojson annotations into .png files. We just used the ROI images, not the context images. For the ease of use, we reuploaded the dataset in the google-drive folder (https://drive.google.com/drive/folders/1enKbMiYK7gnbsL2nn06R146XTFo5Uq46). Due to the license, we cannot share the processed pre-training dataset.

##### 3. Pre-process the pre-training dataset

1. Run experiment planning: `nnUNetv2_plan_and_preprocess -d 998` (998 is the PUMA tissue dataset)
2. Extract the dataset fingerprint of the pre-training dataset: `nnUNetv2_extract_fingerprint -d 999` (999 is the NSCLC dataset)
3. Change the plans: Instead of the suggested plans, we changed the following settings in the `nnUNetPlans.json` for the PUMA dataset (identifier 998, `/nnUNet_preprocessed/Dataset998_Melanom/nnUNetPlans.json`):
   1. Set the batch_size to 12,
   2. Set the patch-size to 512 x 512 instead of 1024 x 1024
4. Transfer the plans: `nnUNetv2_move_plans_between_datasets -s 998 -t 999 -sp nnUNetPlans -tp nnUNetPlans`
5. Run the preprocessing on the pretraining dataset: `nnUNetv2_preprocess -d 999 -plans_name nnUNetPlans`


For more information, check out the pretraining instructions here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md

##### 4. Add a new trainer class to the nnUNet
Ass the NSCLC dataset has regions that are not annotated and should be ignored, we used a slighlty adapted trainer that uses the Cross-Entropyloss with ignore-index class:

```python
class CrossEntropyLossIgnoreBase(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(ignore_index=12)
    def forward(self, input, target):
        # Squeeze the input tensor along the specified dimension
        input_tensor = input[0]
        target_tensor = target[0].squeeze()
        # Call the superclass forward function with modified input
        return super().forward(input_tensor, target_tensor.long())

class nnUNetTrainerIgnoreIndex(nnUNetTrainer):
    def _build_loss(self):
        loss = CrossEntropyLossIgnore()

        return loss
```
We already included the changes in the installed version

##### 5. Perform pretraining
Pretraining command: `nnUNetv2_train -tr nnUNetTrainerIgnoreIndexBase 999 2d 0`

##### 6. Perform finetuning: 
Finetune the network using the best checkpoint from pretraining:
`nUNetv2_train 998 2d 0 -pretrained_weights /path/to/pretrained/checkpoint/Dataset999_Lung/nnUNetTrainerIgnoreIndexBase__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

Just keep the best checkpoint for the submission.

### CellViT++

1. Follow the installation instruction for the CellViT++ algorithm presented here: `https://github.com/TIO-IKIM/CellViT-plus-plus`. For simplicity, we added the code in the cellvitpp-train folder as well. To make the model work, download the CellViT-SAM-H-x40-AMP.pth checkpoint from the [GDrive](https://drive.google.com/drive/folders/1enKbMiYK7gnbsL2nn06R146XTFo5Uq46) (placed in the cellvit++ folder)

2. Download the preprocessed dataset which is already in the required CellViT++ format from the [GDrive](https://drive.google.com/drive/folders/1enKbMiYK7gnbsL2nn06R146XTFo5Uq46) folder (placed under datasets/nuclei). There are both datasets for track 1 and track 2.

3. We performed a Sweep using 100 runs and bayesian optimization to determine the best hyperparameter setting. Based on the sweep results, we retrieved to following optimal configurations:
   1. track-1: [./cellvitpp-train/track-1/training-configuration.yaml](cellvitpp-train/track-1/training-configuration.yaml)
    2. track-2: [./cellvitpp-train/track-2/training-configuration.yaml](cellvitpp-train/track-2/training-configuration.yaml)
4. Please adapt the paths to the datasets and models to your local paths in the .yaml configuration files.

5. The CellViT++ model can be trained using the following command: `python3 cellvit/train_cell_classifier_head.py --config ./cellvitpp-train/track-1/training-configuration.yaml` or `python3 cellvit/train_cell_classifier_head.py --config ./cellvitpp-train/track-2/training-configuration.yaml`. Please be aware to use the correct paths! Use the final checkpoints for each track