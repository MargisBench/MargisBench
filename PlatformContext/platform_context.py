from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

from typing import Dict, Any, Tuple, Union, Optional
from pandas import DataFrame
from torch.utils.data import DataLoader
from pathlib import Path

from BenchmarkingFactory.aiModel import AIModel
from Utils.utilsFunctions import pickAPlatform, acceleratorWarning, initialPrint

class PlatformContext():


    def __init__(self) -> None:
        """
        Initialize the Device Context of the system.
        
        This constructor helps to initialize the system, 
        depending from the underlying hardware platform (Generic, Coral, Fusion) 
        and creates the specific strategy objects (Runner, Initializer, Manager) required 
        to operate on that hardware.
        """

        self.__packageDownloadManager = None
        self.__runnerModule = None
        self.__configurationManager = None
        self.__statsModule = None
        self.__initializer = None
        self.__plotter = None
        self.__platform = pickAPlatform()

        match self.__platform:

            case "generic":
                
                # --- Generic (ONNX) Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerGeneric 
                from ConfigurationModule.configurationManager import ConfigManagerGeneric 
                from PlatformInitializers.initializer import GenericInitializer
                from Runner.runner import RunnerModuleGeneric
                from Utils.plotter import PlotterGeneric

                self.__configurationManager = ConfigManagerGeneric(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerGeneric()
                self.__initializer = GenericInitializer()
                self.__runnerModule = RunnerModuleGeneric()
                self.__plotter = PlotterGeneric()

                logger.debug(f"CONTEXT INITIALIZED:")
                logger.debug(f"RUNNER MODULE: GENERIC RUNNER with {self.__runnerModule}")

            case "coral":

                # --- Coral Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerCoral
                from ConfigurationModule.configurationManager import ConfigManagerCoral
                from PlatformInitializers.initializer import CoralInitializer
                from Runner.runner import RunnerModuleCoral
                from Utils.plotter import PlotterCoral

                acceleratorWarning()

                self.__configurationManager = ConfigManagerCoral(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerCoral()
                self.__initializer = CoralInitializer()
                self.__runnerModule = RunnerModuleCoral()
                self.__plotter = PlotterCoral()

            case "fusion_844_ai":

                # --- Fusion Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerFusion
                from ConfigurationModule.configurationManager import ConfigManagerFusion
                from PlatformInitializers.initializer import FusionInitializer
                from Runner.runner import RunnerModuleFusion
                from Utils.plotter import PlotterFusion

                acceleratorWarning()

                self.__configurationManager = ConfigManagerFusion(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerFusion()
                self.__initializer = FusionInitializer()
                self.__runnerModule = RunnerModuleFusion()
                self.__plotter = PlotterFusion()
            
            case _:
                logger.error(f"No Match for platform")
                exit(0)



    def run(self, aimodel: AIModel, input_data: DataLoader, config_id: str) -> Dict[str, Any]:
        """
        Function that delegates the inference execution to the specific platform Runner.

        Parameters
        ----------
        - aimodel: AIModel
          The model object to run inference on.
        - input_data: DataLoader
          The input dataset for the inference session.
        - config_id: str
          The unique configuration identifier for this run.

        Returns
        -------
        - stats: dict
          Dictionary containing inference statistics (e.g., Latency, FPS, Accuracy).
        """
        return self.__runnerModule._runInference(aimodel=aimodel, input_data=input_data, config_id=config_id)

    def initializePlatform(self, config: Dict[str, Any], config_id: str) -> None:
        """
        Function that delegates platform-specific setup (directories, dependencies) to the Initializer.

        Parameters
        ----------
        - config: dict
          The configuration dictionary containing setup requirements.
        - config_id: str
          The unique ID for the current configuration.
        """
        initialPrint('PLATFORM INITIALIZATION')
        self.__initializer.setConfig(config)
        self.__initializer.setConfigID(config_id)
        self.__initializer.initialize()

    def createConfigFile(self, config: Dict[str, Any]) -> str:
        """
        Function that creates the configuration file via the ConfigurationManager.

        Parameters
        ----------
        - config: dict
          The configuration data to serialize.

        Returns
        -------
        - config_id: str
          The hash/ID generated for the created configuration file.
        """
        return self.__configurationManager.createConfigFile(config)

    def loadConfigFile(self, config_path=None)-> Tuple[Dict[str, Any], str]:
        """
        Function that loads an existing configuration file via the ConfigurationManager.

        Returns
        -------
        - result: tuple
          A tuple containing the configuration dictionary and its config_id string.
        """
        if config_path:
          return self.__configurationManager.loadConfigFile(config_path)
        else: 
          return self.__configurationManager.loadConfigFile()
    
    def checkDownloadedDependencies(self) -> None:
        """
        Function that verifies if necessary platform dependencies (packages, libraries) are installed.
        Delegates to the PackageDownloadManager.
        """
        self.__packageDownloadManager.checkDownloadedDependencies()

    def createPlots(self, df: DataFrame, save_path: Union[str, Path]) -> None:
        """
        Function that delegates plot generation to the platform-specific Plotter.

        Parameters
        ----------
        - df: pandas.DataFrame
          The DataFrame containing benchmark results.
        - save_path: str or Path
          The directory path where plots should be saved.
        """
        self.__plotter.create_plots(df, save_path)


    def getPlatform(self):
          return self.__platform