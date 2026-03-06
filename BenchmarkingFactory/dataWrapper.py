from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch 
import random
from pathlib import Path
from importlib import import_module
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from Utils.utilsFunctions import getModelTransforms
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataWrapper():

    def __init__(self) -> None:
        """
        Constructor to create a DataWrapper for handling dataset loading and processing

        Parameters
        ----------
        None
            Initializes internal loader placeholders to None. 
            Data is not loaded until loadInferenceData() is called.
        """
        self.fine_tuning_loader= None
        self.inference_loader = None
        self.current_data_config = None


    def getCalibrationLoader(self, num_samples: int=100) -> Optional[DataLoader]:
        """
        Function that generates a subset DataLoader for quantization calibration

        Parameters
        ----------
        - num_samples: int, optional (default=100)
          The number of random samples to include in the calibration dataset

        Returns
        -------
        - calibration_loader: torch.utils.data.DataLoader
          A DataLoader containing a small, random subset of the test data. 
          Returns None if inference data has not been loaded.
        """

        if self.inference_loader is None:
            logger.warning("Data has not been loaded... call loadInferenceData first")
            return None

        full_dataset = self.inference_loader.dataset

        total_samples = len(full_dataset)
        num_samples = min(num_samples, total_samples)
        
        indices = random.sample(range(total_samples), num_samples)
        calibration_subset = Subset(full_dataset, indices)

        return DataLoader(
            calibration_subset, 
            batch_size=self.current_data_config['batch_size'], 
            shuffle=False,
            num_workers=0 # Keep it simple for calibration
        )

    def loadInferenceData(self, dataset_info: Dict[str, Any], model_info: Dict[str, Any]) -> None:
        """
        Function that loads the dataset from the filesystem and sets up DataLoaders

        Parameters
        ----------
        - dataset_info: dict
          Dictionary containing dataset configuration (e.g., 'data_dir', 'batch_size')
        - model_info: dict
          Dictionary containing model specific info used for transforms (e.g., 'image_size', 'weights_class')
        """

        logger.debug(f"-----> [DATAWRAPPER MODULE] LOAD INFERENCE DATA")
        logger.debug(f"Loading dataset from: {dataset_info['data_dir']}.")
        
        image_size = model_info['image_size']
        weights = model_info['weights_class']

        self.current_data_config = {"data_dir": dataset_info['data_dir'], "batch_size": dataset_info['batch_size'], "image_size": model_info['image_size']}

        data_path = PROJECT_ROOT / self.current_data_config['data_dir']
        data_transforms = getModelTransforms(model_info)

        try:
            inference_dataset = datasets.ImageFolder(
                str(data_path / "test"),
                data_transforms
            )

            self.inference_loader = DataLoader(
                inference_dataset, 
                batch_size=dataset_info['batch_size'],
                shuffle=False,
                drop_last=True
            )


            train_dataset = datasets.ImageFolder(
                str(data_path / "train"),
                data_transforms
            )

            self.fine_tuning_loader = DataLoader(
                train_dataset, 
                batch_size=dataset_info['batch_size'],
                shuffle=False
            )

            self.current_data_config['class_names'] = inference_dataset.classes
            logger.info(f"Data loaded and setted on {model_info['model_name']}. Classes found: {self.getDatasetInfo('class_names')}")

        except FileNotFoundError:
            logger.error(f"Data directory not found at {data_path / 'test'}")
            self.inference_loader = None
            self.current_data_config['class_names'] = None
        except Exception as e:
            logger.error(f"An error occurred during data loading: {e}")
            self.current_data_config['class_names'] = None

        logger.debug(f"<----- [DATAWRAPPER MODULE] LOAD INFERENCE DATA\n")
        
    def getLoader(self) -> Optional[DataLoader]:
        """
        Function that returns the currently loaded inference dataloader

        Returns
        -------
        - inference_loader: torch.utils.data.DataLoader
          The DataLoader configured for the 'test' dataset
        """
        if self.inference_loader is None:
            logger.warning("Data has not been loaded... check the loading process")
        return self.inference_loader


    def getFineTuningLoader(self) -> DataLoader:
        """
        Function that returns the currently loaded fine-tuning (train) dataloader

        Returns
        -------
        - fine_tuning_loader: torch.utils.data.DataLoader
          The DataLoader configured for the 'train' dataset
        """
        if self.fine_tuning_loader is None:
            logger.critical("Data has not been loaded... check the loading process")
            exit(0)
        return self.fine_tuning_loader

    def getDatasetInfo(self, info: str) -> Any:
        """
        Function that returns specific configuration info about the loaded dataset

        Parameters
        ----------
        - info: str
          Key name to retrieve from the current_data_config dictionary (e.g., 'class_names', 'batch_size')

        Returns
        -------
        - value: any
          The value associated with the requested key in the configuration
        """

        return self.current_data_config[info]
    


if __name__ == "__main__":


    logger.debug("--- DATAWRAPPER TEST: This is a DEBUG log ---")

    model_path = str(PROJECT_ROOT / "ModelData" / "Weights" / "casting_efficientnet_b0.pth")
    info = {
            'module': 'torchvision.models',
            'model_name': "efficientnet_v2",
            'native': True,
            'distilled': False,
            'weights_path': model_path,
            'device': "cpu",
            'class_name': 'efficientnet_b0',
            'weights_class': 'EfficientNet_B0_Weights.DEFAULT',
            'image_size': 224,
            'num_classes': 2,
            'description': 'efficientnet_v2 from torchvision'
        }

    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    dataset = DataWrapper()
    dataset.loadInferenceData(model_info=info, dataset_info = dataset_info)

    classes = dataset.getDatasetInfo('class_names')
    logger.debug(f"Dataset classes are {classes}")

    logger.debug("--- DATAWRAPPER TEST END ---")




