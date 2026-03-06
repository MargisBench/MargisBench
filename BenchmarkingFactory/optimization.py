from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


import os
import gc
import onnx
import torch
import torch_pruning as tp
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader
from onnxruntime import quantization
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from BenchmarkingFactory.calibrationDataReader import CustomCalibrationDataReader
from BenchmarkingFactory.dataWrapper import DataWrapper
from BenchmarkingFactory.aiModel import AIModel
from PlatformContext.platform_context import PlatformContext
from Utils.utilsFunctions import trainEpoch, checkModelExistence



PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Optimization(ABC): 
    """
    Abstract Base Class for optimization strategies (Pruning, Quantization, Distillation).
    """

    @abstractmethod
    def applyOptimization(self) -> Tuple[object, bool]:
        """
        Abstract method to apply the specific optimization logic.
        Must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def getOptimizationInfo(self, info: str) -> Any:
        """
        Abstract method to retrieve configuration details.
        Must be implemented by concrete subclasses.
        """
        pass

class PruningOptimization(Optimization):

    def __init__(self, optimization_config: Dict[str, Any]):
        """
        Constructor for PruningOptimization
        
        Parameters
        ----------
        - optimization_config: dict
          Configuration dictionary containing pruning parameters (e.g., method, amount, n)
        """
        
        self.current_aimodel = None
        self.optimization_config = optimization_config

    def applyOptimization(self, steps: Optional[int]=None, input_examples: Optional[DataLoader]=None, calibration_loader: Optional[DataLoader]=None, fine_tune_loader: Optional[DataLoader]=None, config_id: Optional[str]=None) -> Tuple[AIModel, bool]:
        """
        Function that applies structural pruning to the attached AIModel

        Parameters
        ----------
        - steps: int
          Number of iterative pruning steps
        - input_examples: torch.utils.data.DataLoader
          DataLoader providing input examples for shape inference and dependency graph construction
        - calibration_loader: torch.utils.data.DataLoader, optional
          (Not used in pruning, but kept for interface consistency)
        - fine_tune_loader: torch.utils.data.DataLoader
          DataLoader used for fine-tuning the model after pruning steps
        - config_id: str
          Unique identifier for the current configuration to check for existing models

        Returns
        -------
        - pruned_aimodel: AIModel
          The resulting pruned AIModel instance
        - already_exists: bool
          True if the optimized model was loaded from cache, False if newly created
        """

        logger.debug(f"-----> [OPTIMIZATION MODULE] APPLY STRUCTURAL PRUNING")


        if not self.current_aimodel:
            raise MissingAIModelError("No AIModel set.")

        #Setting 1 single thread for memory saving
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1" 


        method = self.getOptimizationInfo('method')
        device = self.current_aimodel.getInfo('device')
        amount = self.getOptimizationInfo('amount')

        current_model_info = self.current_aimodel.getAllInfo()
        pruned_model_info = deepcopy(current_model_info)
        pruned_aimodel = AIModel(pruned_model_info)
        pruned_aimodel.getAllInfo()['model_name'] += "_pruned"
        pruned_aimodel.getAllInfo()['description'] += f"(Structurally Pruned amount {amount})"

        if checkModelExistence(pruned_aimodel, config_id):
            return (pruned_aimodel, True)

        model_to_prune = pruned_aimodel.getModel()

       


        #for Fine Tuning
        steps=steps
        #optimizer = optim.SGD([p for p in model_to_prune.parameters() if p.requires_grad], lr=0.01, momentum=0.7)
        optimizer = optim.Adam([p for p in model_to_prune.parameters() if p.requires_grad], lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        #for Fine Tuning

        optimizer.zero_grad(set_to_none=True)

        
        if method != "Random":
            n = self.getOptimizationInfo('n')

        try:
            batch = next(iter(input_examples))
            if isinstance(batch, (list, tuple)):
                example_inputs = batch[0]
            else:
                example_inputs = batch

            example_inputs = example_inputs.to(device)
            logger.debug(f"Shape inference using input size: {example_inputs.shape}")
        except Exception as e:
            logger.error(f"Failed to fetch batch from loader: {e}")

        # Define importance strategy
        imp = self.__getImportanceMethod(method)

        # Ignore last classifier
        ignored_layers = []
        for m in model_to_prune.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.current_aimodel.getInfo('num_classes'):
                ignored_layers.append(m)

        pruner = tp.pruner.MagnitudePruner(
            model_to_prune,
            example_inputs,
            importance=imp,
            iterative_steps= steps,    
            pruning_ratio=amount, 
            round_to=8,  
            ignored_layers=ignored_layers,
        )

        del example_inputs
        gc.collect()
        
        # Execute Pruning
        logger.info(f"Pruning model with structural pruning... Target Sparsity: {amount}. Doing the pruning in {steps} steps...")

        for i in range(steps):
            pruner.step() 
            #Fine Tuning

            trainEpoch(model_to_prune, fine_tune_loader, criterion, optimizer, pruned_model_info["device"])

            optimizer.zero_grad(set_to_none=True)
            
        del pruner
        del optimizer
        del criterion
        gc.collect()

        # Check the result
        logger.debug(f"Model physically shrunk. New structure applied.")

        logger.info(f"PRUNING APPLIED WITH {method}, on {amount*100}% of the nodes on {current_model_info['model_name']}")

        
        
        logger.debug(f"<----- [OPTIMIZATION MODULE] DONE")

        return pruned_aimodel, False

    def __getImportanceMethod(self, method: str) -> Any:
        """
        Internal function to get the Importance method class from torch_pruning

        Parameters
        ----------
        - method: str
          The name of the importance method (e.g., "Random", "LnStructured")

        Returns
        -------
        - importance_method: torch_pruning.importance.Importance
          The importance strategy object
        """

        if method == "Random":
            logger.info(f"Using Random Strategy for Pruning")
            imp = tp.importance.RandomImportance()
        elif method == "LnStructured":
            logger.info(f"Using Ln Structuerd Strategy for Pruning")
            n = self.getOptimizationInfo('n')
            imp = tp.importance.MagnitudeImportance(p=n)

        return imp


    def setOptimizationConfig(self, optimization_config: Dict[str, Any]) -> None:
        """
        Function that sets the optimization configuration

        Parameters
        ----------
        - optimization_config: dict
          Configuration dictionary containing pruning parameters
        """
        self.optimization_config = optimization_config

    def setAIModel(self, AIModel: AIModel) -> None:
        """
        Function that sets the AIModel target for optimization

        Parameters
        ----------
        - AIModel: AIModel
          The AIModel object to apply pruning on
        """
        self.current_aimodel =  AIModel

    def getOptimizationInfo(self, info: str) -> Any:
        """
        Function that returns specific info from the optimization config

        Parameters
        ----------
        - info: str
          Key to retrieve from the config dictionary

        Returns
        -------
        - value: any
          Value associated with the key
        """
        return self.optimization_config[info]


class QuantizationOptimization(Optimization):

    def __init__(self, optimization_config: Dict[str, Any]):
        """
        Constructor for QuantizationOptimization

        Parameters
        ----------
        - optimization_config: dict
          Configuration dictionary containing quantization parameters (e.g., bit_method)
        """
        
        self.current_aimodel = None
        self.optimization_config = optimization_config

    def getOptimizationInfo(self, info: str) -> Any:
        """
        Function that returns specific info from the optimization config

        Parameters
        ----------
        - info: str
          Key to retrieve from the config dictionary

        Returns
        -------
        - value: any
          Value associated with the key
        """
        return self.optimization_config[info]

    def setOptimizationConfig(self, optimization_config: Dict[str, Any]) -> None:
        """
        Function that sets the optimization configuration

        Parameters
        ----------
        - optimization_config: dict
          Configuration dictionary containing quantization parameters
        """
        self.optimization_config = optimization_config

    def setAIModel(self, AIModel: AIModel) -> None:
        """
        Function that sets the AIModel target for optimization

        Parameters
        ----------
        - AIModel: AIModel
          The AIModel object to apply quantization on
        """
        self.current_aimodel =  AIModel

    def applyOptimization(self, steps: Optional[int]=None, input_examples: Optional[DataLoader]=None, calibration_loader: Optional[DataLoader]=None, fine_tune_loader: Optional[DataLoader]=None, config_id=None) -> Tuple[AIModel, bool]:
        """
        Function that applies static quantization to the attached AIModel using ONNX Runtime

        Parameters
        ----------
        - calibration_loader: torch.utils.data.DataLoader
          DataLoader used for calibrating quantization parameters
        - config_id: str
          Unique identifier for saving/loading ONNX models

        Returns
        -------
        - quantized_aimodel: AIModel
          The resulting quantized AIModel instance
        - already_exists: bool
          True if the optimized model was loaded from cache, False if newly created
        """
            
        current_model_info = self.current_aimodel.getAllInfo()
        quantized_model_info = deepcopy(current_model_info)
        quantized_aimodel = AIModel(quantized_model_info)
        quantized_aimodel.getAllInfo()['model_name'] += "_quantized"
        quantized_aimodel.getAllInfo()['description'] += f"(Statically Quantized with bit_method:{self.getOptimizationInfo('method')})"


        if checkModelExistence(quantized_aimodel, config_id):
            return (quantized_aimodel, True)

        
        logger.debug(f"-----> [OPTIMIZATION MODULE] APPLY QUANTIZATION OPTIMIZATION")

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["ONNXRUNTIME_INTRA_OP_NUM_THREADS"] = "1"
        os.environ["ONNXRUNTIME_INTER_OP_NUM_THREADS"] = "1"
    
        if not self.current_aimodel:
            raise MissingAIModelError(
                "No AIModel set. Call setAIModel() before applying optimization"
            )

        gc.collect()

        device = self.current_aimodel.getInfo('device')
        model_name = self.current_aimodel.getInfo('model_name')
        bit_method = self.getOptimizationInfo('method')

        

        logger.debug(f"Creating a copy of the model {current_model_info['model_name']}")

        
        res = self.__staticQuantizationOnnx(model_name, device, calibration_loader, bit_method, config_id)

        
        logger.debug(f"<----- [OPTIMIZATION MODULE] APPLY QUANTIZATION OPTIMIZATION")

        return quantized_aimodel, True

    def __staticQuantizationOnnx(self, model_name: str, device: str, inputs: DataLoader, bit_method: str, config_id: str) -> bool:
        """
        Internal function to perform Static Quantization on the ONNX model

        Parameters
        ----------
        - model_name: str
          Name of the model to locate files
        - device: str
          Target device ('cpu' or 'gpu') to determine symmetric/asymmetric quantization
        - inputs: DataLoader
          Calibration data
        - bit_method: str
          Quantization method (e.g., 'QInt8')
        - config_id: str
          Configuration ID for file paths

        Returns
        -------
        - success: bool
          True if quantization succeeded, False otherwise
        """  

        model_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}"  / f"{model_name}.onnx")
        model_prep_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" / f"{model_name}_prep.onnx")
        model_quantized_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" /f"{model_name}_quantized.onnx")
        weight_type = getattr(quantization.QuantType, bit_method)
        onnx.shape_inference.infer_shapes_path(model_path, model_prep_path)

        # Cleaning memory after inference shape
        gc.collect()

        quantization.shape_inference.quant_pre_process(
            model_prep_path, 
            model_prep_path, 
            skip_symbolic_shape=False
        )

        #q_format = quantization.QuantFormat.QOperator
        q_format = quantization.QuantFormat.QDQ

        q_static_opts = None

        if device == "gpu" or device == "cuda":
            q_static_opts = {
                "ActivationSymmetric": True,
                "WeighSymmetric": True
            }
        elif device == "cpu":
            q_static_opts = {
                "ActivationSymmetric": False,
                "WeightSymmetric": True
            }

        # Calibration dataset
        qdr = CustomCalibrationDataReader(inputs, input_name = "input")

        logger.info(f"Starting Static Quantization | {bit_method} (QOperator)...")

        quantized_model = quantization.quantize_static(
            model_input=model_prep_path,
            model_output=model_quantized_path,
            calibration_data_reader=qdr,
            quant_format = q_format,
            reduce_range = True,
            per_channel=True,
            weight_type = weight_type,
            activation_type= quantization.QuantType.QUInt8,
            calibrate_method = CalibrationMethod.Entropy,
            extra_options=q_static_opts
        )

        logger.info("Quantization Complete")

        del qdr
        gc.collect()

        if quantized_model is not None:
            return True
        else:
            return False


class DistillationOptimization(Optimization):

    def __init__(self, optimization_config: Dict[str, Any]):
        """
        Constructor for DistillationOptimization

        Parameters
        ----------
        - optimization_config: dict
          Configuration dictionary containing paths to distilled weights
        """
        
        self.current_aimodel = None
        self.optimization_config = optimization_config

    def applyOptimization(self, steps: Optional[int] = None, input_examples: Optional[DataLoader] = None, calibration_loader: Optional[DataLoader] = None, fine_tune_loader: Optional[DataLoader] = None, config_id: Optional[str] = None) -> Tuple[AIModel, bool]:        
        """
        Function that replaces the model weights with a pre-trained distilled version

        Parameters
        ----------
        - config_id: str, optional
          (Not strictly used in this method but kept for interface consistency)

        Returns
        -------
        - distilled_aimodel: AIModel
          The new AIModel instance loaded with distilled weights
        - already_exists: bool
          Always False currently as it creates a new object instance
        """

        logger.debug(f"-----> [OPTIMIZATION MODULE] APPLY DISTILLATION OPTIMIZATION")
    
        if not self.current_aimodel:
            raise MissingAIModelError(
                "No AIModel set. Call setAIModel() before applying optimization"
            )

        current_model = self.getAIModel()
        current_info = current_model.getAllInfo()
        model_name = current_info['model_name']
        num_classes = current_info['num_classes']
        custom_weights_path = self.getOptimizationInfo('distilled_paths')[model_name]


        # Creating new info to pass to the new model
        distilled_info = deepcopy(current_info)
        distilled_info['model_name'] = current_info['model_name'] + "_distilled"
        distilled_info['weights_path'] = custom_weights_path 
        
        logger.info(f"CREATING STRUCTURE FOR {model_name}_distilled...")


        # Create new model with correct distilled weights
        distilled_aimodel = AIModel(distilled_info)

        logger.debug(f"<----- [OPTIMIZATION MODULE] APPLY DISTILLATION OPTIMIZATION")

        return distilled_aimodel, False
        

    def getOptimizationInfo(self, info: str) -> Any:
        """
        Function that returns specific info from the optimization config

        Parameters
        ----------
        - info: str
          Key to retrieve from the config dictionary

        Returns
        -------
        - value: any
          Value associated with the key
        """
        return self.optimization_config[info]

    def setAIModel(self, model: AIModel) -> None:
        """
        Function that sets the current AIModel for distillation

        Parameters
        ----------
        - model: AIModel
          The target AIModel object
        """

        self.current_aimodel = model

    def getAIModel(self) -> AIModel:
        """
        Function that returns the currently set AIModel

        Returns
        -------
        - current_aimodel: AIModel
        """

        return self.current_aimodel


class MissingAIModelError(Exception):
    """
    Raised when an optimization is applied before an AIModel is set.
    """

    pass

if __name__ == "__main__":


    # Model Example
    model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"

    # Dataset Example
    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }
    dataset = DataWrapper()



    # -------- TEST WITH BASE MODEL ---------------

    context = PlatformContext('generic')

    efficient_path = str(model_weights_path / "casting_efficientnet_b0.pth")
    config_id = "TEST_CONFIG_ID"

    # Faking user input
    efficient_info = {
        'module': 'torchvision.models',
        'model_name': "efficientnet_b0",
        'native': True,
        'weights_path': efficient_path,
        'device': "cpu",
        'class_name': 'efficientnet_b0',
        'weights_class': 'EfficientNet_B0_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'efficientnet_v2 from torchvision'
    }

    mobile_path = str(model_weights_path / "mobilenet_v2.pth")
    mobile_info = {
        'module': 'torchvision.models',
        'model_name': "mobilenet_v2",
        'native': True,
        'weights_path': mobile_path,
        'device': "cpu",
        'class_name': 'mobilenet_v2',
        'weights_class': 'MobileNet_V2_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'mobilenet_v2 from torchvision'

    }

    mnas_path = str(model_weights_path / "mnasnet1_0.pth")
    mnas_info = {
        'module': 'torchvision.models',
        'model_name': "mnasnet1_0",
        'native': True,
        'weights_path': mnas_path,
        'device': "cpu",
        'class_name': 'mnasnet1_0',
        'weights_class': 'MNASNet1_0_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'mnasnet_v2 from torchvision'

    }

    # Creating base model
    efficientnet = AIModel(efficient_info)
    logger.debug(f"Inference with {efficientnet.getInfo('model_name')}, description: {efficientnet.getInfo('description')}")
    logger.debug(f"Test inference on casting product")


    # Attaching dataset to unpruned model
    dataset.loadInferenceData(model_info = efficientnet.getAllInfo(), dataset_info = dataset_info)
    inference_loader = dataset.getLoader()

    # Inference with unpruned model
    efficientnet.createOnnxModel(inference_loader, config_id)
    efficient_stats = context.run(aimodel=efficientnet, input_data=inference_loader, config_id=config_id)

    # Base model for mobilenet
    mobilenet = AIModel(mobile_info)
    mobilenet_dataset = DataWrapper()
    mobilenet_dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
    mobilenet_loader = mobilenet_dataset.getLoader()
    mobilenet.createOnnxModel(mobilenet_loader, config_id)
    mobilenet_stats = context.run(aimodel=mobilenet, input_data=mobilenet_loader, config_id=config_id)

    # Base model for mnasnet
    mnasnet = AIModel(mnas_info)
    mnasnet_dataset = DataWrapper()
    mnasnet_dataset.loadInferenceData(model_info = mnas_info, dataset_info = dataset_info)
    mnasnet_loader = mnasnet_dataset.getLoader()
    mnasnet.createOnnxModel(mnasnet_loader, config_id)
    mnas_stats = context.run(aimodel=mnasnet, input_data=mnasnet_loader, config_id=config_id)


    # # -----------------------------------------------

    # -------- TEST WITH PRUNED MODEL -------------------
    #Pruning Info example
    pruning_info = {
        "method": "LnStructured",
        "n": 1,
        "amount": 0.2
    }

    # Creating Pruning Optimization Object
    pruning_optimizator = PruningOptimization(pruning_info)
    
    # Creating new Pruned AIModels
    pruning_optimizator.setAIModel(efficientnet)
    pruned_efficientnet = pruning_optimizator.applyOptimization(inference_loader)

    pruning_optimizator.setAIModel(mobilenet)
    pruned_mobilenet = pruning_optimizator.applyOptimization(mobilenet_loader)

    pruning_optimizator.setAIModel(mnasnet)
    pruned_mnasnet = pruning_optimizator.applyOptimization(mnasnet_loader)


    # Inference with prunde model
    pruned_efficientnet.createOnnxModel(inference_loader, config_id)
    context.run(aimodel=pruned_efficientnet, input_data=inference_loader, config_id=config_id)

    pruned_mobilenet.createOnnxModel(mobilenet_loader, config_id)
    context.run(aimodel=pruned_mobilenet, input_data=mobilenet_loader, config_id=config_id)

    pruned_mnasnet.createOnnxModel(mnasnet_loader, config_id)
    context.run(aimodel=pruned_mnasnet, input_data=mnasnet_loader, config_id=config_id)

    # # -----------------------------------------------

    # # ---------- TEST WITH QUANTIZED MODEL ---------------

    # quantization_info = {
    #     "arch": "x86", #arch
    #     "method": "QInt8",
    #     "type": "static"
    # }

    # quantization_optimizator = QuantizationOptimization(quantization_info)
    
    # # Creating new quantized AIModel
    # quantization_optimizator.setAIModel(efficientnet)
    # quantized_efficientnet = quantization_optimizator.applyOptimization(inference_loader)

    # # Attaching dataset to quantized model
    # dataset.loadInferenceData(model_info = quantized_efficientnet.getAllInfo(), dataset_info = dataset_info)
    # inference_loader = dataset.getLoader()

    # quantized_efficientnet.runInference(inference_loader)

    # quantization_optimizator.setAIModel(mobilenet)
    # quantized_mobilenet = quantization_optimizator.applyOptimization(mobilenet_loader)
    # mobilenet_dataset.loadInferenceData(model_info = quantized_mobilenet.getAllInfo(), dataset_info = dataset_info)
    # mobilenet_loader = mobilenet_dataset.getLoader()
    # quantized_mobilenet.runInference(mobilenet_loader)


    # -----------------------------------------------

    # ---------- TEST WITH DISTILLED MODEL ---------------

    distillation_info = {
        'method': True,
        'distilled_paths': {
            'mobilenet_v2': '/home/salvatore/Desktop/MasterThesis/MasterThesisTests/BenchProto/ModelData/Weights/mobilenet_v2_distilled.pth',
            'efficientnet_b0': '/home/salvatore/Desktop/MasterThesis/MasterThesisTests/BenchProto/ModelData/Weights/efficientnet_b0_distilled.pth'
        }
    }

    distillation_optimizator = DistillationOptimization(distillation_info)
    distillation_optimizator.setAIModel(efficientnet)  
    distilled_efficientnet = distillation_optimizator.applyOptimization()  

    distilled_efficientnet.createOnnxModel(inference_loader, config_id)
    context.run(aimodel=distilled_efficientnet, input_data=inference_loader, config_id=config_id)

    # -----------------------------------------------




