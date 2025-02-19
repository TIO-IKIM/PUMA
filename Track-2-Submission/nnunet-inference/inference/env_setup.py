import os 
from dotenv import load_dotenv

def setup_nnunet():
    load_dotenv()
    nnUNet_raw = os.getenv("nnUNet_raw")
    nnUNet_preprocessed = os.getenv("nnUNet_preprocessed")
    nnUNet_results = os.getenv("nnUNet_results")
