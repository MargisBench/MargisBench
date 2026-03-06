import os
import sys
import torch
import numpy as np
import random
from importlib import import_module
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mnasnet1_0, MNASNet1_0_Weights,  MobileNet_V2_Weights, mobilenet_v2 #CHANGE HERE
from torchvision.transforms import Compose, Resize, CenterCrop, PILToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# --- CONFIGURATION ---
NUM_SAMPLES = 200 
BLUE = "\x1b[34m"
RESET = "\x1b[0m"

def generate_calibration_data(model_name: str, model_weights: str, batch_size: str, image_size: str, dataset_path: str) -> None:
    """
    This function generates the calibration data for the specified model. Each .npy file will be saved in ./DatasetArrays.

    Parameters
    ----------
    - model_name: str
    The model name.
    - model_weights: str
    The model weights classin str format.
    - batch_size: 
    The batch_size in str format.
    - image_size:
    The image_size in str format.
    - dataset_path:
    The dataset path. 

    Returns
    -------
    - None

    """
    BATCH_SIZE = int(batch_size)
    IMAGE_SIZE = int(image_size)

    # Getting the Image Transformations (Normalization, Resize etc) from officials
    module = import_module("torchvision.models")
    weight_class = getattr(module, model_weights)
    weights = getattr(weight_class, 'DEFAULT')

    #official_transforms = weights.transforms()

    transform_config = weights.transforms()
    resize_size = transform_config.resize_size
    crop_size = transform_config.crop_size
    interpolation = transform_config.interpolation

    calibration_transform = Compose([
        Resize(resize_size, interpolation=interpolation),
        CenterCrop(crop_size),
        PILToTensor()
    ])

    seed = int(os.getenv("SEED"))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=calibration_transform)
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
    OUTPUT_PATH = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "Calibration" / "CalibrationArrays" / f"{model_name}_calibration_data.npy")

    old_umask = os.umask(0)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True, mode=0o777)
    os.umask(old_umask)

    np.save(OUTPUT_PATH, calibration_array)
    print(f"["+ BLUE + "INFO" + RESET + "]" + f"SUCCESS! Final .npy shape {calibration_array.shape}, Saved to: {OUTPUT_PATH}\n")
    

if __name__ == "__main__":
    """
    The main function takes the parameters from the CLI and calls the 
    generate_calibration_data function.

    """

    args = sys.argv[1:]
    model_name = args[0]
    model_weights = args[1]
    batch_size = args[2]
    image_size = args[3]
    dataset_path = args[4]

    generate_calibration_data(model_name, model_weights, batch_size, image_size, dataset_path)
