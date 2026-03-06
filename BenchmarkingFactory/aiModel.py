
from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


from typing import Any
import torch
import torch.nn as nn
import onnxruntime as ort
from numpy import float32
from os import remove, mkdir, getpid, umask
from importlib import import_module
#from psutil import Process
from pathlib import Path
from torchvision import models
from torch.utils.data import DataLoader
from BenchmarkingFactory.dataWrapper import DataWrapper
#from PlatformContext.platform_context import PlatformContext
from Utils.utilsFunctions import getHumanReadableValue
from Utils.calculateStats import CalculateStats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AIModel():

    def __init__(self, model_info: dict):
        
        """
        Constructor to create an AIModel used in a DoE experiment

        Parameters
        ----------
        - model_info: dict
        A dictionary that holds all the information of an AIModel:
            - "module": Library to get the Original architecture of the model
            - "model_name": Name of the model used in the framework for various thing
            - "native": Nature of the model (native = originally included in the framework)
            - "weights_path": Weights path of the pre-trained model
            - "device": Device used for generic Inference
            - "class_name": Class of the model from module
            - "weights_class": Class of the weights, used to load pretrained weights
            - "image_size": Input size and Size of the compatible images with the model
            - "num_classes": Output size of the model
            - "task": Description of the task performed by the model
            - "description": Brief description of the model
        """

        self.__model_info = model_info
        self.__model = self._loadModel(model_info['module'], model_info['class_name'])

    def getAllInfo(self) -> dict:
        """
        Function that returns the entire dict of AIModel info

        Returns
        -------
        - model_info : dict (to major info check init of AIModel)
        """
        return self.__model_info

    def getInfo(self, info_name: str) -> Any:
        """
        Function that returns one of the info in AIModel info

        Returns
        -------
        - specific_model_info : any (to major info check init of AIModel)
        """
        return self.__model_info[info_name]


    def _replaceModelClassifier(self, model: nn.Module, class_name: str) -> None:
        """
        Function that automatically recognize the final layer and replace for the inference on a specific dataset

        Parameters
        ----------
        - model: torchvision model where to replace the Last classifier
        - class_name: model class of the torchvision model
        """

        logger.debug(f"-----> [AIMODEL MODULE] REPLACE MODEL CLASSIFIER")

        # Inferencing the classifier of the model, changing it
        if self.getInfo('num_classes') is not None:
            last_layer_name, last_layer = None, None

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    last_layer_name = name
                    last_layer = module

            if last_layer is None:
                logger.error(f"Could not find a Linear layer to replace in {class_name}")
            else:
                # Get in_features and create new layer
                try:
                    in_features = last_layer.in_features
                    num_classes = self.getInfo('num_classes')
                    new_classifier = torch.nn.Linear(in_features, num_classes)
                    
                    # Split the name to navigate it and replace last layer
                    parts = last_layer_name.split('.')
                    if len(parts) == 1:
                        # Simple case: 'fc'
                        setattr(model, last_layer_name, new_classifier)
                    else:
                        # Nested case: 'classifier.1' or heads.head
                        parent_module = model
                        for part in parts[:-1]:
                            # Handle cases where parent is a Sequential list
                            if part.isdigit():
                                parent_module = parent_module[int(part)]
                            else:
                                parent_module = getattr(parent_module, part)
                    
                        # Set the last part
                        last_part_name = parts[-1]
                        if last_part_name.isdigit():
                            parent_module[int(last_part_name)] = new_classifier
                        else:
                            setattr(parent_module, last_part_name, new_classifier)
                    
                    logger.debug(f"Replace {last_layer_name} for {num_classes} classes.")

                except Exception as e:
                    logger.error(f"Failed to auto-replace classifier for {class_name}: {e}")
        logger.debug(f"<----- [AIMODEL MODULE] REPLACE MODEL CLASSIFIER")

    def _loadModel(self, module: str, class_name: str) -> nn.Module: 
        """
        Function to Load the torchvision model from module and the class name

        Parameters
        ---------- 
        - module: Module from PyTorch Library (e.g., torchvision.models)
        - class_name: Attribute name of the module, to load the model structure

        Returns
        -------
        - model: torchvision model
        """
        
        logger.debug(f"-----> [AIMODEL MODULE] LOAD MODEL")

        module = import_module(module)
        model_class = getattr(module, class_name)
        model = model_class()

        # Try to replace last layer classifier
        self._replaceModelClassifier(model, model_class)


        try:

            weights = self.getInfo('weights_path')
            model_checkpoint = torch.load(weights, map_location=torch.device('cpu'))

            # Check if it's a checkpoint dictionary or a raw state_dict
            if 'model_state_dict' in model_checkpoint:
                model.load_state_dict(model_checkpoint['model_state_dict'])
                logger.debug(f"Loaded Weights from model_state_dict key in {weights}")
            
            # Assuming it's a raw state_dict
            else:
                model.load_state_dict(model_checkpoint)
                logger.debug(f"Loaded Weights raw state_dict from {weights}")

            logger.info(f"BASIC STRUCTURE MODEL LOADED CORRECTLY: {class_name}")

        except FileNotFoundError:
            logger.error(f"Weight file not found at {self.getInfo('weights_path')}. Model has random weights.")
        except Exception as e:
            logger.error(f"Error loading model weights from {self.getInfo('weights_path')}: {e}")

        logger.debug(f"<----- [AIMODEL MODULE] LOAD MODEL\n")

        return model

    def _getProviderList(self, device: str) -> list:
        """
        Function that scans possible providers and give the right provider list for the inference session

        Parameters
        ---------- 
        - device: a string given from the configurator module (e.g., 'cpu' or 'gpu')

        Returns
        -------
        - provider_list: list of possible providers ordered in terms of priority
        """

        machine_provider_list = ort.get_available_providers()
        choosen_device_provider_list = []


        if device == "cpu":
            choosen_device_provider_list.extend([
                "OpenVINOExecutionProvider", 
                "ArmNNExecutionProvider", 
                "ACLExecutionProvider"
            ])
        elif device == "gpu":
            choosen_device_provider_list.extend([
                "TensorRTExecutionProvider", 
                "CUDAExecutionProvider", 
                "ROCMExecutionProvider"
            ])

        
        # Always put the cpu provider at the end as a fallback
        choosen_device_provider_list.append("CPUExecutionProvider")

        # Intersect the wanted providers with actual available providers
        final_provider_list = [
            provider for provider in choosen_device_provider_list 
            if provider in machine_provider_list
        ]

        if not final_provider_list:
            logger.warning(f"Provider not correctly choosed, setting default cpu provider")
            return ["CPUExecutionProvider"]

        return final_provider_list



    def setModel(self, model: nn.Module) -> None: 
        """
        Set new model to the AIModel

        Parameters
        ----------
        - model: new Pytorch nn module 
        """

        self.__model = model

    def getModel(self) -> nn.Module:
        """
        Returns the torch model attached to aimodel

        Returns
        -------
        - model: torchvision model currently attached to AIModel
        """

        return self.__model

    def createOnnxModel(self, input_data: DataLoader, config_id: str) -> None:
        """
        Function that creates the onnx file to use for inferencing

        Parameters
        ----------
        - input data: dataloader with test dataset
        - config_id: ID of the chosen configuration
        """

        logger.debug(f"-----> [AIMODEL MODULE] CREATE ONNX MODEL")


        model_name = self.getInfo('model_name')
        onnx_directory_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" 
        onnx_model_path = onnx_directory_path /f"{model_name}.onnx"


        if not onnx_directory_path.exists():
            old_umask = umask(0)
            mkdir(onnx_directory_path, mode=0o777)
            umask(old_umask)

        if onnx_model_path.exists():
            logger.info(f"ONNX FILE OF {model_name} ALREADY EXISTS AT {onnx_model_path}")
            return #TO PASS THE CREATION IF IT ALREADY EXISTS 

        # Getting parameters
        onnx_model_path = str(onnx_model_path)
        device = self.getInfo('device')
        image_size = self.getInfo('image_size')

        self.getModel().to(device)
        self.getModel().eval()

        dummy_input = None

        try:
            inputs, _ = next(iter(input_data))

            dummy_input = inputs.to(device)

            current_batch_size = dummy_input.shape[0]
            logger.debug(f"Generated dummy input from real dataset (sliced to batch size {current_batch_size})")

        except (StopIteration, TypeError, AttributeError):
            logger.warning("Could not fetch data from loader. Generating random dummy input.")
            dummy_input = torch.randn(32, 3, image_size, image_size).to(device)

        # Dynamic axes config (Standard exporter)
        # dynamic_axes_config = {
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'}
        # }

        try:
            torch.onnx.export(
                self.getModel(),
                dummy_input,
                onnx_model_path,
                export_params=True,
                opset_version=13,
                do_constant_folding = False,
                input_names = ['input'],
                output_names = ['output'],
                #dynamic_axes = dynamic_axes_config,
                dynamo = False
            )
            logger.info(f"ONNX Model successfully created at {onnx_model_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")

        logger.debug(f"<----- [AIMODEL MODULE] CREATE ONNX MODEL\n")


    

# if __name__ == "__main__":

#     logger.debug("----------------- AI MODULE TEST: This is a DEBUG log --------------------")

#     context = PlatformContext()

#     model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"
#     efficient_path = str(model_weights_path / "casting_efficientnet_b0.pth")

#     config_id = "TEST_CONFIG_ID"
    
#     efficient_info = {
#         'module': 'torchvision.models',
#         'model_name': "efficientnet_v2",
#         'native': True,
#         'weights_path': efficient_path,
#         'device': 'cpu',
#         'class_name': 'efficientnet_b0',
#         'weights_class': 'EfficientNet_B0_Weights.DEFAULT',
#         'image_size': 224,
#         'num_classes': 2,
#         'description': 'efficientnet_v2 from torchvision'
#     }

#     mobile_path = str(model_weights_path / "mobilenet_v2.pth")
#     mobile_info = {
#         'module': 'torchvision.models',
#         'model_name': "mobilenet_v2",
#         'native': True,
#         'weights_path': mobile_path,
#         'device': 'cpu',
#         'class_name': 'mobilenet_v2',
#         'weights_class': 'MobileNet_V2_Weights.DEFAULT',
#         'image_size': 224,
#         'num_classes': 2,
#         'description': 'mobilenet_v2 from torchvision'

#     }

#     mnas_path = str(model_weights_path / "mnasnet1_0.pth")
#     mnas_info = {
#         'module': 'torchvision.models',
#         'model_name': "mnasnet1_0",
#         'native': True,
#         'weights_path': mnas_path,
#         'device': 'cpu',
#         'class_name': 'mnasnet1_0',
#         'weights_class': 'MNASNet1_0_Weights.DEFAULT',
#         'image_size': 224,
#         'num_classes': 2,
#         'description': 'mnasnet1_0 from torchvision'

#     }

#     data_dir = "ModelData/Dataset/casting_data"

#     dataset_info = {
#         'data_dir': data_dir,
#         'batch_size': 32
#     }

#     efficientnet = AIModel(efficient_info)
#     mobilenet = AIModel(mobile_info)
#     mnasnet = AIModel(mnas_info)
#     dataset = DataWrapper()
    
#     efficient_name = efficientnet.getInfo("model_name")
#     mobile_name = mobilenet.getInfo("model_name")
    
#     # First Inference test: with efficientnet
#     dataset.loadInferenceData(model_info = efficient_info, dataset_info = dataset_info)
#     inference_loader = dataset.getLoader()
#     efficientnet.createOnnxModel(inference_loader, config_id)
#     #efficient_stats = efficientnet.runInference(inference_loader, config_id)
#     efficient_stats = context.run(aimodel=efficientnet, input_data=inference_loader, config_id = config_id)

#     # Second Inference test: with mobilenet
#     dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
#     inference_loader = dataset.getLoader()
#     mobilenet.createOnnxModel(inference_loader, config_id)
#     #mobile_stats = mobilenet.runInference(inference_loader, config_id)
#     mobile_stats = context.run(aimodel=mobilenet, input_data=inference_loader, config_id=config_id)


#     # Third Inference test: with mnasnet
#     dataset.loadInferenceData(model_info = mnas_info, dataset_info = dataset_info)
#     inference_loader = dataset.getLoader()
#     mnasnet.createOnnxModel(inference_loader, config_id)
#     mnas_stats = context.run(aimodel=mnasnet, input_data=inference_loader, config_id=config_id)
#     #mnas_stats = mnasnet.runInference(inference_loader, config_id)

#     logger.debug("------------- AI MODULE TEST END -------------------")
