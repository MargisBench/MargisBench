import os
import sys
import torch
import numpy as np
import random
from importlib import import_module
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mnasnet1_0, MNASNet1_0_Weights,  MobileNet_V2_Weights, mobilenet_v2 #CHANGE HERE
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Any, Union

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent



# --- CONFIGURATION ---
NUM_SAMPLES = 200 
BLUE = "\x1b[34m"
RESET = "\x1b[0m"

def generate_calibration_data(model_name: str, model_weights: str, batch_size: Any, image_size: Any, dataset_path: str) -> None:

    """
    Function that generates a calibration dataset stored as a .npy file.
    
    This function loads a subset of the dataset, applies the correct preprocessing transforms 
    derived from the model weights (Resize, Crop), and saves the images as a NumPy array.
    
    Critically, it transposes the tensor format from PyTorch standard (NCHW) to 
    TensorFlow/TFLite/Hailo standard (NHWC) before saving, which is required for 
    the quantization compilers of these platforms.

    

    Parameters
    ----------
    - model_name: str
      The name of the model (used for naming the output file).
    - model_weights: str
      The class name of the weights (e.g., 'MobileNet_V2_Weights') to dynamically load transforms.
    - batch_size: str or int
      The batch size to use during data loading.
    - image_size: str or int
      The target input size (unused in logic as transforms define it, but passed for consistency).
    - dataset_path: str
      The root directory of the dataset.
    """

    BATCH_SIZE = int(batch_size)
    IMAGE_SIZE = int(image_size)

    # Getting the Image Transformations (Normalization, Resize etc) from officials
    module = import_module("torchvision.models")
    weight_class = getattr(module, model_weights)
    weights = getattr(weight_class, 'DEFAULT')


    transform_config = weights.transforms()

    seed = int(os.getenv("SEED"))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform_config)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"["+ BLUE + "INFO" + RESET + "]" + f"FOUND {len(dataset)} IMAGES. COLLECTING {NUM_SAMPLES} SAMPLES...")
    all_images = []
    

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, total=NUM_SAMPLES, desc="["+ BLUE + "INFO" + RESET + "]" + " CREATING")):
            if i >= NUM_SAMPLES:
                break
            
            # image shape: (1, 3, 224, 224) -> NCHW
            numpy_batch = images.cpu().numpy()
            
            # TRANSPOSE to NHWC: (1, 3, 224, 224) -> (1, 224, 224, 3)
            # This is critical for TFLite/onnx2tf
            nhwc_batch = numpy_batch.transpose(0, 2, 3, 1)
            
            all_images.append(nhwc_batch)
            

    # CONCATENATE into one 4D array
    # Resulting shape: (NUM_SAMPLES, 224, 224, 3)
    calibration_array = np.concatenate(all_images, axis=0)


    # Saving the array
    OUTPUT_PATH = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "CalibrationArrays" / f"{model_name}_calibration_data.npy")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    np.save(OUTPUT_PATH, calibration_array)
    print(f"["+ BLUE + "INFO" + RESET + "]" + f"SUCCESS! Final .npy shape {calibration_array.shape}, Saved to: {OUTPUT_PATH}\n")
    

if __name__ == "__main__":

    # Get model name and weights from terminal
    args = sys.argv[1:]
    model_name = args[0]
    model_weights = args[1]
    batch_size = args[2]
    image_size = args[3]
    dataset_path = args[4]

    generate_calibration_data(model_name, model_weights, batch_size, image_size, dataset_path)
