from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


from jsonschema import validate, ValidationError
from json import load, dump, decoder
from rich.pretty import pprint
from os.path import exists
from os import listdir
from pathlib import Path
from numpy import delete
from pathlib import Path
from hashlib import sha512
from platform import uname
from abc import ABC, abstractmethod
from datetime import datetime
from Utils.utilsFunctions import getLongestSubString, getFilenameList, initialPrint
from typing import Dict, List, Any, Optional, Tuple, Union



PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "config.json") #config file path
config_schemas_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "Schemas") #configSchema file path
config_history_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "configHistory.json") #configHistory file path
models_library_path= str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "models_library.json") #models_library file path
optimizations_library_path = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "optimizations_library.json") #optimizations_library file path
models_weights_path = str(PROJECT_ROOT / "ModelData" / "Weights") #weights of embedded models in the framework
VALID_CHOICES = {'y','n'} #Choices for CPU Usage
OPTIMIZATIONS_NEED_ARCH = {"Quantization"} #Optimizations that needs the arch type of the system.
OPTIMIZATION_NEED_N = {"LnStructured"} #Optimizations methods that needs the 'n' parameter
BLUE = "\x1b[34m"
RESET = "\x1b[0m"


error_dataset_path_message =""" 
ModelData/Dataset/
└── casting_data
    ├── test
    │   ├── def_front
    │   └── ok_front
    └── train
        ├── def_front
        └── ok_front
"""

class ConfigManager(ABC):

    def __init__(self, platform: str):
        """
        Creates the ConfigManger object. It sets two protected variables: _platform and _arch. 
        _arch variable refers to the architecture where the whole framework is executed, so, it can mismatch from
        the target device architecture.

        Parameters
        ----------
        - platform: str
        The target platform choosen by the user.  
        
        """

        try:
            self._platform= platform
            self._arch = uname().machine
        except (FileNotFoundError, Exception) as e:
            logger.error(f"Encountered a generic problem initializatin the ConfigManager.\nThe specific error is: {e}.")



    def _printConfigFile(self, input: Any, topic: str) -> None:
        """
        Prints the input variable with pprint along a specified topic.

        Parameters
        ----------
        - input: Dict
        The struct to print (it can be a Dict or a List).

        - topic: str 
        The specific topic to print. 

        Returns
        -------
        - None

        """
        print("\n" +"-"*10 + '\x1b[32m' + topic + '\033[0m' + "-"*10)
        pprint(input, expand_all=True)
        print("-"*10 + "-"*len(topic)+"-"*10+"\n")


    
    def _checkModels(self, models: List[Dict[str, Union[str, int, bool]]]) -> bool:
        """
        Checks the availability of models wrote in config file and applies the needed changes.
        (e.g. if a model is not available in the 'models_library.json', that specific model
        is removed from the config and an error message is shown to warn the user)

        Parameters
        ----------

        - models: List 
        The list of chosen models gathered from the configuration file. 

        Returns
        -------
        - result: bool
        The boolean result about the Models section check.

        """
        
        try:
            models_library = None
            changed=False
            idx_to_del=[]

            try:
                with open(models_library_path, "r") as models_library_file:
                    models_library = load(models_library_file)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False


            #make a set of value in order to improve the performance in searching.
            models_library_sets = {
                key: set(value) for key, value in models_library.items()
            }


            for idx, model in enumerate(models):
                if model["native"]:

                    if model["model_name"] not in models_library_sets:
                        logger.error(f"The model {model['model_name']} is not present in the model library. Removing it from the config...\n")
                        idx_to_del.append(idx)
                        changed=True
                        continue

                    logger.info(f"CHANGING CONFIG TO NATIVE {model['model_name']} MODEL...")
                    models[idx]=models_library[model["model_name"]]
                    changed= True

                else:  #checks the custom model
                    logger.info(f"CHECKING CONFIG FOR CUSTOM {model['model_name']} MODEL...")

                    if exists(model["weights_path"]):
                        logger.info(f"CHECKING CONFIG WEIGHT PATH FILE FOR {model['model_name']}...")
                    else:
                        logger.error(f"There are no weights file for {model['model_name']}. Try to provide it in ./ModelData/Weights/ dir or check the weights path in config files.\n")
                        return False


            if changed:

                models[:] = delete(models, idx_to_del).tolist() #Deleting the "native" models not present in models_library
                if len(models)==0:
                    logger.critical("NO MODEL PRESENT IN THE CONFIGURATION. EXITING...")
                    return False

                logger.info(f"SHOWING NEW MODELS CONFIGURATION...")
                self._printConfigFile(models, " MODELS SECTION ")

            else:
                logger.info(f"CONFIGURATION NOT CHANGED...")
                self._printConfigFile(models, " MODELS SECTION ")


        except (Exception) as e:
            logger.error(f"Encountered a generic problem in model check.\nThe specific error is: {e}.\n")
            return False
        return True


    @abstractmethod
    def _checkOptimizations(self, optimizations: Dict[str, Dict[str, Any]], model_dicts: List[Dict[str, Union[str, int, bool]]]) -> bool:
        """
        Checks the availability of optimization methods wrote in config file. 
        The specific implementation changes due to specific platform (Quantization allowed/not allowed).

        Parameters
        ----------
        - optimizations: Dict 
        Dict of chosen optimizations from the configuration file.
        - model_dicts:   List
        List of chosen models from the configuration file.
    
        Returns
        -------
        - result: bool
        The boolean result about the Optimizations section check.

        """
        pass


    def _checkDataset(self, dataset_dict: Dict[str, Union[str, int]]) -> bool:
        """
        Checks if the dataset path specified contains at least two directories (Kaggle Standard). 
        The validity of the dataset will be checked later. 

        Parameters
        ----------
        - dataset_dict: Dict
        The dict of dataset section specified in the config file.

        Returns
        -------
        - result: bool
        The boolean result about the Dataset section check.

        """

        dataset_path = dataset_dict["data_dir"] + "/test"

        logger.info(f"CHECKING DATASET PATH...")
        if exists(dataset_path) and len(listdir(dataset_path))>1:
            logger.info(f"DATASET PATH RECOGNISED!")
            self._printConfigFile(dataset_dict, " DATASET SECTION ")
            return True
        
        logger.error(f"Dataset path not recognised! You should have a similar path configuration (with at least two classes):")
        print(error_dataset_path_message)

        return False

    
    def _updateConfigHistory(self, config: Dict[str, Union[List, Dict, int]], hash_value: str) -> None:
        """
        This function asks to the user if the loaded/created configurations has to be added to the historyConfig.json file. 
        If the hash_value (key) is already present, the function returns.

        Parameters
        ----------

        - config: Dict
        The created/loaded configuration.
        
        - hash_value: str 
        The hash value created from the configuration file (config_id).
        
        Returns
        -------
        - None
        
        """

        history_dict = {}
        with open(config_history_path, "r") as config_history_file:
            try:
                history_dict = load(config_history_file)
                if hash_value in history_dict.keys():
                    logger.info("THE CONFIG. IS ALREAY PRESENT INTO THE HISTORY!")
                    return

            except decoder.JSONDecodeError as e:
                logger.info(f"THE HISTORY FILE WAS EMPTY!")


        while True:
            choice = input(f"["+ BLUE + "INFO" + RESET + "]" + " DO YOU WANT TO SAVE THE CONFIG. INTO THE CONFIG. HISTORY? (y/n): ").lower()

            if choice in VALID_CHOICES:
                if choice == 'y':
                    try:

                        history_dict[f"{hash_value}"] = config

                        with open(config_history_path, "w") as config_history_file:
                            dump(history_dict, config_history_file, indent=4)

                        logger.info("CONFIG. CORRECTLY ADDED TO THE HISTORY!")

                    except (FileNotFoundError,Exception) as e:
                        logger.error(f"Encountered a problem saving the config in the history.\nThe specific error is: {e}.\n")
            else:
                logger.error("Invalid Input. Please enter 'y' or 'n'.")
                continue
            break
        print("\n")


    def _addArchType(self, config: Dict[str, Union[List, Dict, int, str]]) -> None:

        """
        Adds the architecture type where the framework is executed ('x86' or 'aarch') to the config file. 

        Parameters
        ----------
        - config: Dict
        The config dict.
        
        Returns
        -------
        - None

        """
        config["arch"] = self._arch


    def _addPlatform(self, config: Dict[str, Union[List, Dict, int]]) -> None:
        """
        Adds the choosen platform by the user to the config file. 

        Parameters
        ----------
        - config: Dict
        The config dict.
        
        Returns
        -------
        - None

        """

        config["platform"] = self._platform




    def _createDistilledPaths(self, optimizations: Dict[str, Any], model_dicts: List[Dict[str, Union[str, int, bool]]]) -> None:
        """
        Creates the distilled paths for loading the distilled version of the chosen models.

        Parameters
        ----------

        - optimizations: Dict 
        Dict of optimizations from the configuration.

        - model_dicts: List
        List of all models choosen from the configuration.
        
        Returns
        -------
        - None

        """

        file_name_list = getFilenameList(models_weights_path)
        file_name_list = [file_name for file_name in file_name_list if "_distilled" in file_name]

        optimizations['Distillation']['distilled_paths'] = {}

        for model_dict in model_dicts:

            best_candidate_for_model, best_file_name = "", ""
            for file_name in file_name_list:

                file_name_without_pth = file_name.removesuffix("_distilled.pth")

                best_candidate_for_name = getLongestSubString(model_dict['model_name'], file_name_without_pth)
                best_candidate_for_path = getLongestSubString(model_dict['weights_path'].split("/")[-1].removesuffix(".pth"), file_name_without_pth)

                best_candidate_between_name_path = max(best_candidate_for_name, best_candidate_for_path, key=len)

                if len(best_candidate_between_name_path) > len(best_candidate_for_model):
                    best_candidate_for_model = best_candidate_between_name_path
                    best_file_name = file_name

                # Perfect Match, found the distilled weights
                if len(file_name_without_pth) == len(best_candidate_for_model):
                    break

            if len(best_file_name.removesuffix("_distilled.pth")) == len(best_candidate_for_model):
                logger.info(f"MODEL: {model_dict['model_name']} | YOU FOUND THE CORRECT DISTILLED MODEL {best_file_name}")
            elif len(best_file_name) == 0:
                logger.critical(f"MODEL: {model_dict['model_name']} | NO MATCH WITH NONE FILE FOR DISTILLED MODEL")
                exit(0)
            else:
                logger.warning(f"MODEL: {model_dict['model_name']} | YOU FOUND A PARTIAL MATHC FOR A DISTILLED MODEL: {best_file_name}")

            optimizations['Distillation']['distilled_paths'][model_dict['model_name']] =  f"{models_weights_path}/{best_file_name}"           


    def loadConfigFile(self, path : Optional[str]=config_path ) -> Tuple[Dict[str, Union[List, Dict, int]], str]:
        """
        Loads the configuration from a JSON file (ConfigFiles/config.json). 

        Parameters
        ----------
        - path: str
        The path of the configuration file. It should be a JSON file. The default one is (./ConfigFiles/config.json) 
        
        Returns
        -------
        - config: Dict 
        The configuration dict.
        - hash_value: str
        The hash value generated on the config file.

        """
        initialPrint("CONFIGURATION FILE")
        config = ""
        try:
            with open(path, "r") as config_file:
                config = load(config_file)
                logger.info("LOADING CONFIGURATON...")

        
            logger.info("VALIDATING LOADED CONFIGURATION...")
            validate(instance=config, schema=self._schema)

        except (ValidationError, Exception) as e:
            logger.critical(f"Encountered a problem validating the config file. Check if the fields provided are correct.\nThe specific error is: {e}.\n")
            exit(0)
    
        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")
        self._printConfigFile(config, " INITIAL CONF. FILE ")

        logger.info("CHECKING THE MODELS...")
        
        if self._checkModels(config["models"]) and self._checkDataset(config["dataset"]) and self._checkOptimizations(config["optimizations"], config["models"]):
            logger.info("DONE!")
            self._addPlatform(config)
            self._addArchType(config)

            self._printConfigFile(config, " FINAL CONF. FILE ")
            hash_value = str(sha512(str(config).encode("utf-8")).hexdigest())[:12] + "_" + self._platform
            self._updateConfigHistory(config, hash_value)
            #Arch for Quantization Optimization
            return config, hash_value
        else:
            logger.critical(f"Something went wrong in the configuration check. Exiting...\n")
            exit(1)

    
    def createConfigFile(self, config: Dict[str, Union[List, Dict, int]]) -> str:
        """
        Creates the configuration file from a constructed dict created by the interactive CLI session or created in the Python Code.

        Parameters
        ----------
        - config: Dict 
        The config dict generated from interactive sessionor code.

        Returns
        -------
        - hash_value: str
        The hash value generated on the config file.
        
        """

        initialPrint("CONFIGURATION FILE CHECKING\n")

        try:
            # It's an useless check, but we'll never know!
            logger.info("VALIDATING CREATED CONFIGURATON...")
            validate(instance=config, schema=self._schema)
        except (ValidationError, Exception) as e:
            logger.critical(f"Encountered a problem validating the config file. Check if the fields provided are correct.\nThe specific error is: {e}.\n")
            exit(0)

        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")

        if self._checkModels(config["models"]) and self._checkDataset(config["dataset"]) and self._checkOptimizations(config["optimizations"], config['models']):
            logger.info("DONE!")
            
            with open(config_path, "w") as config_file:
                dump(config, config_file, indent=4)

            self._addPlatform(config)
            self._addArchType(config)

            self._printConfigFile(config, " FINAL CONF. FILE ")
            logger.info(f"SAVING IT INTO {config_path}...")

            # with open(config_path, "w") as config_file:
            #     dump(config, config_file, indent=4)

            logger.info(f"SAVED!")

            
            hash_value = str(sha512(str(config).encode("utf-8")).hexdigest())[:12] + "_" + self._platform

            self._updateConfigHistory(config, hash_value)
            return hash_value
        else:
            logger.critical(f"Something went wrong in the configuration check. Exiting...\n")
            exit(1)

   


class ConfigManagerGeneric(ConfigManager):

    def __init__(self, platform: str):
        """
        Creates the ConfigMangerGeneric object. It sets three  protectedvariables: _platform and _arch and _schema. 
        _arch variable refers to the architecture where the whole framework is executed, so, it can mismatch from
        the target device architecture.
        _schema variable is the specific loaded schema for the specific platform.

        Parameters
        ----------
        - platform: str
        The target platform choosen by the user.  
        
        """

        super().__init__(platform)
        schema_path = Path(config_schemas_path) / "configSchemeGeneric.json"
        with open(schema_path, "r") as config_schema_file:
            self._schema = load(config_schema_file)


    def _checkOptimizations(self, optimizations: Dict[str, Union[str, int]], model_dicts: List[Dict[str, Union[str, int, bool]]]) -> bool:
        """
        Implementation for Generic Platform. Quantization allowed.
        Checks the availability of optimization methods wrote in config file. 
        The specific implementation changes due to specific platform.

        Parameters
        ----------
        - optimizations: Dict 
        Dict of chosen optimizations from the configuration file.
        - model_dicts:   List
        List of chosen models from the configuration file.
    
        Returns
        -------
        - result: bool
        The boolean result about the Optimizations section check.

        """
        logger.info(f"CHECKING OPTIMIZATION METHODS FOR {self._platform} PLATFORM...")

        try:
            optimizations_library = None
            opt_to_remove=[]

            try:
                with open(optimizations_library_path, "r") as optimizations_library_file:
                    optimizations_library = load(optimizations_library_file)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False

            #make a set of value in order to improve the performance in searching.
            optimizations_library_sets = {
                key: set(value) for key, value in optimizations_library.items()
            }

            for optimization_name in optimizations.keys():

                if optimization_name not in optimizations_library_sets:
                    logger.info(f"THE OPTIMIZATION {optimization_name} IS NOT AVAILABLE. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                elif optimizations[optimization_name]["method"] not in optimizations_library_sets[optimization_name]:
                    logger.error(f"THE OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} DOESN'T EXISTS. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                else:
                    logger.info(f"OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} RECOGNISED!")

                if "n" in optimizations[optimization_name] and optimizations[optimization_name]["method"] not in OPTIMIZATION_NEED_N:
                    optimizations[optimization_name].pop("n")

            if len(opt_to_remove) > 0:
                for name in opt_to_remove:
                    optimizations.pop(name, None)
                    

            if len(optimizations) == 0:
                logger.critical("NO OPTIMIZATIONS PRESENT IN THE CONFIGURATION.")
                return False

            if "Distillation" in optimizations.keys():
                if optimizations['Distillation']['method']:
                    self._createDistilledPaths(optimizations, model_dicts)
            
            self._printConfigFile(optimizations, " OPTIMIZATIONS SECTION ")

        except (Exception) as e:
            logger.error(f"Encountered a generic problem in optimization check.\nThe specific error is: {e}")

        return True



class ConfigManagerCoral(ConfigManager):

    def __init__(self, platform):
        """
        Creates the ConfigMangerCoral object. It sets three  protectedvariables: _platform and _arch and _schema. 
        _arch variable refers to the architecture where the whole framework is executed, so, it can mismatch from
        the target device architecture.
        _schema variable is the specific loaded schema for the specific platform.

        Parameters
        ----------
        - platform: str
        The target platform choosen by the user.  
        
        """
        super().__init__(platform)
        schema_path = Path(config_schemas_path) / "configSchemeCoral.json"
        with open(schema_path, "r") as config_schema_file:
            self._schema = load(config_schema_file)


    def _checkOptimizations(self, optimizations: Dict[str, Union[str, int]], model_dicts: List[Dict[str, Union[str, int, bool]]]) -> bool:
        """
        Implementation for Coral Platform. Quantization not allowed.
        Checks the availability of optimization methods wrote in config file. 
        The specific implementation changes due to specific platform.

        Parameters
        ----------
        - optimizations: Dict 
        Dict of chosen optimizations from the configuration file.
        - model_dicts:   List
        List of chosen models from the configuration file.
    
        Returns
        -------
        - result: bool
        The boolean result about the Optimizations section check.

        """
        logger.info(f"CHECKING OPTIMIZATION METHODS FOR {self._platform} PLATFORM...")

        try:
            optimizations_library = None
            opt_to_remove=[]

            try:
                with open(optimizations_library_path, "r") as optimizations_library_file:
                    optimizations_library = load(optimizations_library_file)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False

            optimizations_library_sets = {}
            #make a set of value in order to improve the performance in searching.
            for key, value in optimizations_library.items():
                if key != "Quantization":
                    optimizations_library_sets[key] = value


            for optimization_name in optimizations.keys():

                if optimization_name not in optimizations_library_sets:
                    logger.info(f"THE OPTIMIZATION {optimization_name} IS NOT AVAILABLE. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                elif optimizations[optimization_name]["method"] not in optimizations_library_sets[optimization_name]:
                    logger.error(f"THE OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} DOESN'T EXISTS. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                else:
                    logger.info(f"OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} RECOGNISED!")

                if "n" in optimizations[optimization_name] and optimizations[optimization_name]["method"] not in OPTIMIZATION_NEED_N:
                    optimizations[optimization_name].pop("n")

            if len(opt_to_remove) > 0:
                for name in opt_to_remove:
                    optimizations.pop(name, None)
                    

            if len(optimizations) == 0:
                logger.critical("NO OPTIMIZATIONS PRESENT IN THE CONFIGURATION.")
                return False

            if "Distillation" in optimizations.keys():
                if optimizations['Distillation']['method']:
                    self._createDistilledPaths(optimizations, model_dicts)
            
            self._printConfigFile(optimizations, " OPTIMIZATIONS SECTION ")

        except (Exception) as e:
            logger.error(f"Encountered a generic problem in optimization check.\nThe specific error is: {e}")

        return True

    

class ConfigManagerFusion(ConfigManager):

    def __init__(self, platform):
        """
        Creates the ConfigMangerFusion object. It sets three  protectedvariables: _platform and _arch and _schema. 
        _arch variable refers to the architecture where the whole framework is executed, so, it can mismatch from
        the target device architecture.
        _schema variable is the specific schema loaded for the specific platform.

        Parameters
        ----------
        - platform: str
        The target platform choosen by the user.  
        
        """

        super().__init__(platform)
        schema_path = Path(config_schemas_path) / "configSchemeFusion.json"
        with open(schema_path, "r") as config_schema_file:
            self._schema = load(config_schema_file)


    def _checkOptimizations(self, optimizations: Dict[str, Union[str, int]], model_dicts: List[Dict[str, Union[str, int, bool]]]) -> bool:
        """
        Implementation for Fusion Platform. Quantization not allowed.
        Checks the availability of optimization methods wrote in config file. 
        The specific implementation changes due to specific platform.

        Parameters
        ----------
        - optimizations: Dict 
        Dict of chosen optimizations from the configuration file.
        - model_dicts:   List
        List of chosen models from the configuration file.
    
        Returns
        -------
        - result: bool
        The boolean result about the Optimizations section check.

        """
        
        return ConfigManagerCoral._checkOptimizations(self, optimizations, model_dicts) #it's quite the same for this device, only quantization is not allowed. 
        

# if __name__ == "__main__":


#     configTest = {
#         "models": [
#             {
#                 "model_name": "mobilenet_v2", 
#                 "native": True
#             },
#             {
#                 "module": "torchvision.models",
#                 "model_name": "efficientnet", 
#                 "native": False,
#                 "distilled": False,
#                 "weights_path": "./ModelData/Weights/casting_efficientnet_b0.pth",
#                 "device": "cpu",
#                 "class_name": "efficientnet_b0",
#                 "weights_class": "EfficientNet_B0_Weights", 
#                 "image_size": 224,
#                 "num_classes": 1000,
#                 "task": "classification",
#                 "description": "EfficientNet from Custom Models"
#             }
#         ],
#         "optimizations": {
#             "Quantization": {
#                 "method": "QInt8",
#                 "type":"static" 
#             },
#             "Pruning": {
#                 "method": "L1Unstructured",
#                 "amount": 0.7
#             },
#             "Distillation": {
#                 'method': True,
#                 'distilled_paths': {}
#             }
#         },
#         "dataset": {
#             "data_dir": "./ModelData/Dataset/casting_data",
#             "batch_size": 32
#         }
#     }


#     configManager = ConfigManager(arch="x86", there_is_gpu=False)

#     #configFile, hash_value = configManager.loadConfigFile()

#     hash_value = configManager.createConfigFile(configTest)
