from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger

import gc
import difflib
import traceback # TRYING
import questionary
import pandas as pd
import numpy as np
import oapackage
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from json import dump
from difflib import SequenceMatcher
from pathlib import Path
from subprocess import run, DEVNULL
from json import dump, load, JSONDecodeError
from statsmodels.graphics.factorplots import interaction_plot
import torch.nn as nn
from tqdm import tqdm
from importlib import import_module
from torchvision import transforms
#from BenchmarkingFactory.aiModel import AIModel
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union



PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_caches_script_bash = "./" / PROJECT_ROOT / "PlatformInitializers/GenericScripts/cleancache.sh"
supported_devices_lib_path = PROJECT_ROOT / "ConfigurationModule/ConfigFiles/supported_devices_library.json"


def compareModelArchitecture(model1: object, model2: object) -> None:
    """ 
    Utility function to compare two model architectures.

    Parameters
    ----------
    
        - model1 : object  
                        The first model object (torch model)
        - model2 : object
                        The second model object (torch model)
    
    Returns
    -------
    
        - None
    """


    model1_str = str(model1).splitlines()
    model2_str = str(model2).splitlines()

    # Create a differ object
    diff = difflib.ndiff(model1_str, model2_str)

    print(f"Comparing Model 1 vs Model 2:")
    print("-" * 30)
    
    # Print only the differences
    has_diff = False
    for line in diff:
        if line.startswith('+') or line.startswith('-'):
            print(line)
            has_diff = True
    
    if not has_diff:
        print("Architectures are identical.")


def getHumanReadableValue(value: bytes, suffix: str="B") -> str:
        """
        Scale bytes to its proper format
        
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'

        Parameters
        ----------
        
        - value : bytes
                    The value in bytes
        - suffix : str
                    The string suffix
        
        Returns
        -------
         
        - string : str
                        The value in string format

        """
        
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if value < factor:
                return f"{value:.2f}{unit}{suffix}"
            value /= factor

def getLongestSubString(string1: str, string2: str) -> str:
    """
    Return the longest substring between two strings. 

    Parameters
    ----------
    
    - string1 : str
                First string.       
    - string2 : str
                Second string.
    
    Returns
    -------
     
    - str : str
                The longest substring.
    """

    string_match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    longest_substring = string1[string_match.a : string_match.a + string_match.size]

    return longest_substring

def getFilenameList(directory_path: str) -> List[str]:
    """
    Return a list of filename in a given directory_path. 

    Parameters
    ----------
     
    - directory_path : str
                    The specified path.

    Returns
    -------
    
    - file_name_list : list 
                    List of filenames.
    """

    file_name_list = []
    dir_path = Path(directory_path)

    for file_path in dir_path.iterdir():

        # Check if it is a file
        if file_path.is_file():
            file_name_list.append(file_path.name)

    if file_name_list:
        return file_name_list
    else:
        raise FileNotFoundError(f"No files were found in this directory: {directory_path}")
        exit(0)

def cleanCaches() -> None:
    """
    This function writes the '3' in /proc/sys/vm/drop_caches file in order to drop the unused pages, inodes and dentries.
    Visit man proc_sys_vm manpages for more.

    Parameters
    ----------
    - None

    Returns
    -------
    - None

    """
    try:

        result = run([str(clean_caches_script_bash)], check=True, stdout=DEVNULL)

        if result.returncode==0:
            logger.info("CACHE CLEANED FOR INDEPENDENT EXPERIMENTS")
    except ChildProcessError as e:
        logger.error(f"Cache not cleaned correctly. The next measurements could be not independent.\nThe error is: {e}")
    except Exception as e:
        logger.error(f"Encountered a generic problem cleaning the caches. The next measurements could be not independent.\nThe error is: {e}")


def subRunQueue(context: object, aimodel: object, inference_loader: DataLoader, config_id: str, queue: object) -> None:
    """
    This function in fundamental to run each inference over different platforms. 
    It use a Queue object to communicate between this and 
    
    Parameters
    ----------
    - context: object
    The subprocess context. (Taken by get_context()) 
    - aimodel: AIModel
    The aiModel object which the nferencing will be performed. 
    - inference_loader: DataLoader
    Torch DataLoader
    - config_id: str
    The configuration id
    - queue: object 
    The queue of the subprocess context. 

    Returns
    -------
    - None

    """
    try:  
        stats = context.run(aimodel=aimodel, input_data=inference_loader, config_id=config_id)

        queue.put({"status": "success", "data": stats})

    except Exception as e:
        logger.error(f"SubProcess CRASHED: {e}")
        logger.error(traceback.format_exc())
        queue.put({"status": "error", "message": str(e)})



def initialPrint(topic: str) -> None:
    """
    Utility function in order to print the main section of the execution in violet. 

    Parameters
    ----------

    - topic: str
    Main section title.

    Returns
    -------
    - None
    """
    print("\n\t\t"+ '\x1b[35m' + topic + '\033[0m')

def trainEpoch(model: object, loader: DataLoader, criterion: object, optimizer: object, device: str) -> float:
    """
    Utility function to perform fine-tuning for one epoch after the Pruning Optimization apply. 


    Parameters
    ----------
    - model: object
    Torch model.
    - loader: DataLoader
    Torch DataLoader.
    - criterion: object
    Torch Criterion. 
    - optimizer: object
    Torch optimizer.
    - device: str
    Target device (cpu, gpu etc.) 

    Returns
    -------
    - result_loss: float
    The loss reached fine-tuning the network for one epoch. 

    """
    model.train()

    #Freezing parameters, unfreezing classifier.
    for param in model.parameters():
            param.requires_grad = False

    # This finds the last linear layer we just added and unfreezes it.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True

    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="[INFO] FINE-TUNING: "):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #nn.functional.dropout(inputs, p=0.5, training=True)
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def checkModelExistence(aimodel: object, config_id: str)-> bool:
    """
    Utility function to check if the ONNX model already exists. 


    Parameters
    ----------
    - aimodel: AIModel
    An AIModel object.
    - config_id: str
    The confguration id. 


    Returns
    -------
    - result: bool
    If the model already exists returns True 

    """

    model_name = aimodel.getAllInfo()['model_name']
    onnx_directory_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" 
    onnx_model_path = onnx_directory_path /f"{model_name}.onnx"

    if onnx_model_path.exists():
        logger.info(f"ONNX file of {model_name} already exists at {onnx_model_path}")
        return True 

    return False


def pickAPlatform() -> Tuple[str]:
    """
    Utility function to allow the user to choose a target platform in an interactive way. 

    Parameters
    ----------
    - None


    Returns
    -------
    - option: str
    The choosen option.

    """

    title = "Choose the target device: "

    try:
        with open(supported_devices_lib_path, "r") as supported_device_lib_file:
            supported_device = load(supported_device_lib_file)


        option = questionary.select(title, choices=supported_device["devices"], pointer='>>',  use_indicator=True).ask()

        if option is None:
            logger.critical(f"None option encountered, exiting...")
            exit(1)

        return option

    except JSONDecodeError as e:
        logger.critical(f"Encountered a problem loading the supported devices library file.\nThe specific problem is {e}")

    except Exception as e:
        logger.critical(f"Encountered a generic proble loading the supported devices library file.\nThe specific problem is: {e}")


    exit(1)

def acceleratorWarning() -> None:
    """
    Utility function to print a warning when the targets provide an accelerator. 

    Parameters
    ----------
    - None

    Returns
    -------
    - None

    """

    logger.warning(f"The target platform is provided with an accelerator.") 
    logger.warning(f"All the base models will be quantized. 'Quantization' optimization field is not allowed.")
    logger.warning("Press a key to continue...")
    input()


def createPathDirectory(path: str) -> None:
    """
    Utility function to create a directory if doesn't exists. 

    Parameters
    ----------
    - path: str
    The target path to check/create.

    Returns
    -------
    - None

    """

    if path.exists():
        if path.is_dir():
            logger.debug("Directory already exists.")
        else:
            logger.debug("Path exists but it is a file, not a directory!")
    else:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory create successfully")
        except PermissionError:
            logger.error("Error: You do not have permission to create this directory")
        except Exception as e:
            logger.error(f"An error occured during the creation of Directory {path}: {e}")


def getModelTransforms(model_info: Dict[str, Union[str, int, bool]]) -> object:
        """
        Internal function that generates the data transforms based on the model's weights class
        
        Parameters
        ----------
        - model_info: dict
          Dictionary containing model details, specifically 'module' and 'weights_class' 
          to dynamically load the correct preprocessing transforms.

        Returns
        -------
        - transforms: torchvision.transforms.Compose
          The composition of transforms required by the pre-trained model
        """

        module = import_module(model_info['module'])

        str_weights = model_info['weights_class']

        # Analyze the weights class, standard weights class aren't more than two parts
        parts = str_weights.split(".")
        weights_class = getattr(module, parts[0])

        if len(parts) == 1:
            weights = getattr(weights_class, "DEFAULT")
        else:
            weights = getattr(weights_class, parts[1])

        #image_size = model_info['image_size']

        return transforms.Compose([
            weights.transforms()
        ])


if __name__ == "__main__":

    base_path = PROJECT_ROOT / "Results" / "ab9066e9f813_generic" / "DoEResults"
    file_path = base_path / "doe_results_raw.csv"
    df = pd.read_csv(file_path)
    time_box_plot(df, base_path)

    

