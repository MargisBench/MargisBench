from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch
import subprocess
import onnxruntime as ort
import numpy as np
import getpass
import socket
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
from numpy import float32
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader
from BenchmarkingFactory.aiModel import AIModel

from BenchmarkingFactory.dataWrapper import DataWrapper
from BenchmarkingFactory.optimization import PruningOptimization
from Utils.utilsFunctions import getModelTransforms, cleanCaches
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class RunnerModule(ABC):
    """
    Abstract Base Class defining the interface for execution runners on different hardware platforms.
    """

    @abstractmethod
    def _runInference(self, aimodel: AIModel, input_data: DataLoader, config_id: str) -> Dict[str, Union[float, str]]:
        """
        Abstract method to execute the inference process. 
        Must be implemented by concrete runner classes (Generic, Coral, Fusion, etc.).

        Parameters
        ----------
        - aimodel: AIModel
          The model object containing metadata and path information.
        - input_data: DataLoader
          The dataset to perform inference on.
        - config_id: str
          The unique configuration ID for the current experiment run.

        Returns
        -------
        - stats: dict
          A dictionary containing calculated statistics (e.g., Latency, FPS, Accuracy).
        """
        pass


class RunnerModuleGeneric(RunnerModule):
    """
    Runner implementation for Generic platforms (CPU/GPU) using ONNX Runtime.
    """

    def __init__(self):
        """
        Constructor for RunnerModuleGeneric.
        """
        pass

    def _runInference(self, aimodel: AIModel, input_data: DataLoader, config_id: str) -> Dict[str, Any]:
        """
        Function to run inference locally on a given dataset using ONNX Runtime.
        Calculates detailed statistics including confusion matrix and profiling data.

        Parameters
        ----------
        - aimodel: AIModel
          The AIModel object containing model metadata and device information.
        - input_data: torch.utils.data.DataLoader
          The dataset loader containing the input images and labels.
        - config_id: str
          The unique identifier for the current configuration run.

        Returns
        -------
        - stats: dict
          A dictionary containing performance metrics (accuracy, loss, latency, FPS, memory usage).
        """
        from Utils.calculateStats import CalculateStatsGeneric

        logger.debug(f"-----> [RUNNER GENERIC MODULE] RUN INFERENCE")

        device_str = aimodel.getInfo('device')
        model_name = aimodel.getInfo('model_name')
        onnx_model_path = PROJECT_ROOT / "ModelData" / "ONNXModels"  / f"{config_id}" / f"{model_name}.onnx"

        provider_list = aimodel._getProviderList(aimodel.getInfo('device'))

        try:
            # Enable profiling
            sess_options = ort.SessionOptions()
            sess_options.enable_mem_pattern = True
            sess_options.enable_profiling = True
            
            sess_options.profile_file_prefix = model_name
            logger.debug(f"Session is enabled with profiling")

            ort_session = ort.InferenceSession(str(onnx_model_path), providers=provider_list, sess_options = sess_options)

            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            input_type = float32
            output_type = float32

            logger.debug(f"Input of ort session: {ort_session.get_inputs()[0]}")
            logger.debug(f"Output of ort session: {ort_session.get_outputs()[0]}")

        except Exception as e:
            logger.debug(f"Error loading ONNX model: {e}")

        io_binding = ort_session.io_binding()

        n_total_images = len(input_data.dataset)
        num_batches = len(input_data)
        logger.info(f"INFERENCING OVER {num_batches} BATCHES...")

        logger.debug(f"In this dataset there are {n_total_images} images across {num_batches} batches")
        
        total = 0
        correct = 0
        running_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        cm = None
        class_names = None

        if hasattr(input_data.dataset, 'classes'):
            class_names = input_data.dataset.classes

        cleanCaches() 
        with torch.no_grad():
            for inputs, labels in tqdm(input_data, desc="[INFO] INFERENCING: "):

                labels = labels.to(device_str)
                batch_size = inputs.shape[0]

                input_as_numpy = inputs.numpy()

                ort_input_value = ort.OrtValue.ortvalue_from_numpy(input_as_numpy)

                # Binding inputs and outputs on cpu
                io_binding.bind_input(
                    name=input_name,
                    device_type='cpu',
                    device_id=0,  
                    element_type=input_type,
                    shape=tuple(input_as_numpy.shape),
                    buffer_ptr=input_as_numpy.ctypes.data 
                )
                io_binding.bind_output(output_name, device_type = 'cpu',
                                        device_id=0)


                ort_session.run_with_iobinding(io_binding)


                # Get outputs and reconvert into torch tensors
                onnx_outputs_ort = io_binding.get_outputs()
                numpy_output = onnx_outputs_ort[0].numpy()
                onnx_outputs_tensor = torch.from_numpy(numpy_output)

                # Cleaning binding for next iteration
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()

                loss = criterion(onnx_outputs_tensor, labels)
                running_loss += loss.item() * batch_size
                predicted_indices = torch.argmax(onnx_outputs_tensor, dim=1)
                total += labels.shape[0]
                correct += (predicted_indices == labels).sum().item()

                if cm is None:
                    num_classes = numpy_output.shape[1]
                    cm = np.zeros((num_classes, num_classes), dtype=int)
                    if not class_names:
                        class_names = [str(i) for i in range(num_classes)]

                # Move to CPU for indexing
                batch_preds = predicted_indices.cpu().numpy()
                batch_labels = labels.cpu().numpy()

                for true_idx, pred_idx in zip(batch_labels, batch_preds):
                    cm[true_idx][pred_idx] +=1

        # Get profile path
        profile_file_path = ort_session.end_profiling()


        if not profile_file_path:
            logger.error(f"Profiling enabled but no file was generated.")
            return {}


        logger.debug(f"Profile file generated: {profile_file_path}")

        if cm is not None:
            CalculateStatsGeneric.printConfusionMatrix(cm, f" {model_name.upper()} CONFUSION MATRIX ", class_names)
        
        # Get kernel stats
        #logger.info(f"MEMORY ALLOCATED FOR THE SESSION: {getHumanReadableValue(memory_after_session-memory_before_session)}")
        #logger.info(f"TOTAL MEMORY ALLOCATED THROUGH RUN (WEIGHTS + ARENA): {getHumanReadableValue(max_memory_arena_allocated)}")
        stats = CalculateStatsGeneric.calculateStats(profile_file_path, num_batches, n_total_images, correct, total, running_loss)

        
        CalculateStatsGeneric.printStats(stats, f" {model_name.upper()} STATS ")            

        logger.debug(f"<----- [AIMODEL MODULE] RUN INFERENCE\n")

        return stats

class RunnerModuleCoral(RunnerModule):
    """
    Runner implementation for Google Coral Edge TPU devices.
    Executes inference via 'mdt' (Mendel Development Tool) commands.
    """

    def __init__(self):
        """
        Constructor for RunnerModuleCoral.
        """
        pass

    def _runInference(self, aimodel: AIModel, input_data: DataLoader, config_id: str) -> Dict[str, Any]:
        """
        Function that executes the inference on a connected Coral device.
        It delegates execution to a remote shell script using 'mdt exec' and pulls results back.

        Parameters
        ----------
        - aimodel: AIModel
          The AIModel object containing model name and metadata.
        - input_data: DataLoader
          The data loader (used here primarily to extract class names and batch count).
        - config_id: str
          The configuration ID used to locate models.

        Returns
        -------
        - stats: dict
          A dictionary containing performance metrics parsed from the Coral device's output CSV.
        """

        from Utils.calculateStats import CalculateStatsCoral

        results_dir = PROJECT_ROOT / "temp_results" / "CoralResults"
        model_name = aimodel.getInfo('model_name')
        edgetpu_name = f"{model_name}_full_integer_quant_edgetpu.tflite"
        num_batches = len(input_data)
        class_list = input_data.dataset.classes
        class_arg = " ".join(class_list)

        os.makedirs(results_dir, exist_ok=True)


        official_transforms = getModelTransforms(aimodel.getAllInfo())
        core_transform = official_transforms.transforms[0]
        logger.debug(f"The transformations are {core_transform}")
        resize_dim = None

        if hasattr(core_transform, 'resize_size'):
            resize_dim = core_transform.resize_size[0]
            logger.debug(f"RESIZE DIM OF {model_name} CATCHED: {resize_dim}")
        else:
            resize_dim = 256
            logger.warning(f"RESIZE DIM OF {model_name} NOT CATCHED, FALLBACK: {resize_dim}")

        logger.debug(f"THESE ARE CLASSES PASSED TO THE SCRIPT: {class_arg}")

        try:
            subprocess.run(
                ["mdt", "exec", "bash", "run_suite.sh", f"./Models/{edgetpu_name}", f"{resize_dim}", class_arg],
                check=True
            )

            subprocess.run(
                ["mdt", "pull", "benchmark_results.csv", str(results_dir)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            logger.error(f"CORAL RUNNER FAILED, Problem with subprocess runner")

        stats = CalculateStatsCoral.calculateStats(str(results_dir / "benchmark_results.csv"), num_batches)

        return stats
        
class RunnerModuleFusion(RunnerModule):
    """
    Runner implementation for Hailo/Fusion AI devices.
    Executes inference via SSH and SCP.
    """

    def __init__(self):
        """
        Constructor for RunnerModuleFusion.
        """
        pass


    def _runInference(self, aimodel: AIModel, input_data: DataLoader, config_id: str) -> Dict[str, Any]:
        """
        Function that executes the inference on a remote Fusion/Hailo device via SSH.
        
        Parameters
        ----------
        - aimodel: AIModel
          The AIModel object containing model name used to locate the HEF file.
        - input_data: DataLoader
          The data loader used to determine batch size and image dimensions.
        - config_id: str
          The unique configuration ID used to structure result directories.

        Returns
        -------
        - stats: dict
          A dictionary containing performance metrics parsed from the remote device's output CSV.
        """
        results_dir = PROJECT_ROOT / "temp_results" / config_id /"FusionResults"
        model_name = aimodel.getInfo('model_name')
        hef_name = f"{model_name}.hef"
        batch_size = input_data.batch_size
        image_size=getModelTransforms(aimodel.getAllInfo()).transforms[0].crop_size[0]
        num_batches = len(input_data)
        user = getpass.getuser()
        hostname = socket.gethostname()
        private_key_path = Path(f'/home/{user}/.ssh/') / "fusion_844_ai"
        destination_host=os.getenv("FUSION_HOST_IP") 

        from Utils.calculateStats import CalculateStatsFusion

        logger.info(f"CREATING {results_dir} FOLDER...")
        os.makedirs(results_dir, exist_ok=True)



        try:
            logger.info(f"STARTING SUITE FOR {model_name} ({hef_name}) MODEL, WITH BATCH SIZE {batch_size}.")
            subprocess.run(
                ["ssh", "-i", private_key_path, '-t', destination_host, f"./run_suite.sh {model_name} ./hef_files/{hef_name} {batch_size} {image_size}"],
                check=True
            )

            logger.info(f"PULLING THE RESULT...")
            subprocess.run(
                ["scp", "-q", "-i", private_key_path, f"{destination_host}:/home/root/benchmark_results.csv", str(results_dir)],
                check=True
            )

            logger.info(f"RESULT FILE FOR {model_name} CORRECTLY PULLED!")


        except subprocess.CalledProcessError as e:
            logger.error(f"Encounterd a CalledProcessError in Fusion Runner.\nThe specific problem is: {e}")


        stats = CalculateStatsFusion.calculateStats(str(results_dir / "benchmark_results.csv"), num_batches)
        
        return stats




if __name__ == "__main__":

    config_id = "6bae1867a5_generic"
    model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"

    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    mnas_path = str(model_weights_path / "mnasnet1_0.pth")
    mnas_info = {
        'module': 'torchvision.models',
        'model_name': "mnasnet1_0",
        'native': True,
        'weights_path': mnas_path,
        'device': 'cpu',
        'class_name': 'mnasnet1_0',
        'weights_class': 'MNASNet1_0_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'mnasnet1_0 from torchvision'

    }

    pruning_info = {
        "method": "LnStructured",
        "n": 1,
        "amount": 0.2
    }

    mnasnet = AIModel(mnas_info)
    mnasnet_dataset = DataWrapper()
    mnasnet_dataset.loadInferenceData(model_info = mnas_info, dataset_info = dataset_info)
    mnasnet_loader = mnasnet_dataset.getLoader()

    # Creating Pruning Optimization Object
    pruning_optimizator = PruningOptimization(pruning_info)
    pruning_optimizator.setAIModel(mnasnet)
    pruned_mnasnet, _ = pruning_optimizator.applyOptimization(1, mnasnet_loader, mnasnet_loader, config_id)


    runner = RunnerModuleCoral()
    runner._runInference(mnasnet, None, None)
    runner._runInference(pruned_mnasnet, None, None)
