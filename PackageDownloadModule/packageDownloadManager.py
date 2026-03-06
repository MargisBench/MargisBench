from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


import os
import venv
from rich.pretty import pprint
from pathlib import Path
from json import load, decoder, dump
from subprocess import check_call, CalledProcessError
from sys import executable
from importlib.metadata import distributions
from time import sleep
from Utils.utilsFunctions import initialPrint
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
requirements_file_directory = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory")
converters_file_directory = str(PROJECT_ROOT / "Converters")
requirements_installed_path= str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / ".installed.json" )



class PackageDownloadManager(ABC):

    @abstractmethod
    def __init__():
        """
        Abstract initializer for PackageDownloadManager.
        It creates the protected variable _builder, that is an EnvBuikder Object.

        """

        self._builder=venv.EnvBuilder(with_pip=True)
    
    @abstractmethod
    def _checkAlreadyInstalled(self) -> (bool, bool):
        """
        Abstract method useful for checking if the dependencies are already installed. 
        The implementation differs based on the platform.
        
        """
        pass
    
    @abstractmethod
    def _downloadDependencies(self, platform: str, installed_requirements_dict: Dict[str, bool]) -> None:
        """
        Asbtract method to download the missing dependencies.
        The implementation differs based on the platform.

        Parameters
        ----------
        - device: str
        The specific target platform.

        - installed_requirements_dict: Dict
        The .installed.json dict object.

        Returns
        -------
        - None

        """
        pass


    def checkDownloadedDependencies(self) -> None:
        """
        Checks if dependecies are already installed calling _checkingAlreadyInstalled. 
        If the dependencies are needed, it will start _downloadDependencies function to download them for the specific platform.


        Parameters
        ----------
        - self

        Output:
        - None
        
        """

        initialPrint("DEPENDENCIES DOWNLOAD\n")
        installed_requirements_dict={}

        try:
            with open(requirements_installed_path, "r") as installed_requirements_file:
                installed_requirements_dict = load(installed_requirements_file)


            installed, _ = self._checkAlreadyInstalled()
            
            if not installed:
                self._downloadDependencies(self._platform, installed_requirements_dict)
            else:
                logger.info(f"NEEDED DEPENDENCIES ALREADY PRESENT...")

            if installed_requirements_dict and not installed:
                with open(requirements_installed_path, "w") as installed_requirements_file:
                    dump(installed_requirements_dict, installed_requirements_file, indent=4)
                
            logger.info("ALL DEPENDENCIES INSTALLED! IF THERE ARE PROBLEMS, MAKE A FORCE-REINSTALL OF THE DEPENDENCIES WITHOUT PIP CACHING.")


        except decoder.JSONDecodeError as e:
            logger.critical(f"Encountered an error Decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
            exit(0)
        except (FileNotFoundError,Exception) as e:
            logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            exit(0)

        


class PackageDownloadManagerGeneric(PackageDownloadManager):

        def __init__(self):
            """
            Creates the PackageDownloadManagerGeneric object for Generic Platform. It will set two protected parameters: _platform and _deps_dir.
            _platform regards the str for the specific platform. 
            _deps_dir regards the path where the dependencies.txt files are present.

            """
            self._platform = "generic"
            self._deps_dir = Path(requirements_file_directory) / "Generic"


        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
            Checks if dependecies are already installed for the Generic Platform, inspectionating the .installed.json file. 

            Parameters
            -----------
            - self
                
            Returns
            -------
            - install_needed: bool
            Boolean variable to install the needed dependencies. (These Dependencies are specified in self._deps_dir)
            - install_gpu: bool (For Future Exetensions)

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)



        def _downloadDependencies(self, platform: str, installed_requirements_dict: Dict[str, bool]) -> None:
            """
            Downloads the dependencies for Generic platform. All the Generic dependencies should be already installed
            with the initialization of the project. In Generic/generic.txt should be added additional dependencies
            needed only with Generic Platform.

            Parameters
            ----------
            - platform: str
            The specific target platform.

            - installed_requirements_dict: Dict
            The .installed.json dict object.

            Returns
            -------
            - None

            """

            requirements_file_generic_path = str(self._deps_dir / "generic.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} DEPENDENCIES...")
                sleep(1)
                return_value = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_generic_path])
                
                if return_value == 0:
                    installed_requirements_dict[platform] = True

            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)


class PackageDownloadManagerCoral(PackageDownloadManager):


        def __init__(self):
            """
            Creates the PackageDownloadManagerCoral object for Coral Platform. It will set five protected parameters: _platform, _deps_dir, _converter_dir, 
            _converter_venv_dir, _builder.
            
            _platform regards the str for the specific platform. 
            _deps_dir regards the path where the dependencies.txt files are present.
            _converter_dir for the CoralConverter dir
            _converter_venv_dir for CoralConverter virtual envirnoment dir
            _builder for the venv builder.

            """
            self._platform = "coral"
            self._deps_dir = Path(requirements_file_directory) / "Coral"
            self._converter_dir = Path(converters_file_directory) / "CoralConverter"
            self._converter_venv_dir = Path(self._converter_dir) / "venv"
            self._builder = venv.EnvBuilder(with_pip=True) #for venv



        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
            Checks if dependecies are already installed for the Coral Platform, inspectionating the .installed.json file. 

            Parameters
            -----------
            - self
                
            Returns
            -------
            - install_needed: bool
            Boolean variable to install the needed dependencies. (These Dependencies are specified in self._deps_dir)
            - install_gpu: bool (For Future Exetensions)

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)


        def _downloadDependencies(self, platform: str, installed_requirements_dict: Dict[str, bool]) -> None:
            """
            Downloads the dependencies for Coral platform. The ones stored in Coral/coral_converter.txt
            are specific for Converters/CoralConverter virtual environment.

            Parameters
            ----------
            - platform: str
            The specific target platform.

            - installed_requirements_dict: Dict
            The .installed.json dict object.

            Returns
            -------
            - None

            """

            requirements_file_coral_path = str(self._deps_dir / "coral.txt")
            requirements_file_coral_converter_dependencies_path = str(self._deps_dir / "coral_converter.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} BASIC DEPENDENCIES...")
                sleep(1)
                return_value_basic = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_coral_path])

                logger.info(f"INSTALLED BASIC DEPENDENCIES! PASSING TO {platform.upper()} CONVERTER ONES...")

                self._builder.create(self._converter_venv_dir)

                logger.info(f"VIRTUAL ENV FOR {self._platform} CONVERTER CREATED. INSTALLING DEPENDENCIES...\n")

                return_value_converter = check_call([os.path.join(self._converter_venv_dir, "bin", "python3.10"),"-m", "pip", "install", "-r", requirements_file_coral_converter_dependencies_path])

                logger.info("INSTALLED DEPENDENCIES IN VENV!")

                if return_value_basic == 0 and return_value_converter == 0:
                    installed_requirements_dict[platform] = True


            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)
                


class PackageDownloadManagerFusion(PackageDownloadManager):


        def __init__(self):
            """
            Creates the PackageDownloadManagerFusion object for Fusion Platform. It will set five protected parameters: _platform, _deps_dir, _converter_dir, 
            _converter_venv_dir, _builder.

            _platform regards the str for the specific platform. 
            _deps_dir regards the path where the dependencies.txt files are present.
            _converter_dir for the CoralConverter dir
            _converter_venv_dir for CoralConverter virtual envirnoment dir
            _builder for the venv builder.

            """
            self._platform = "fusion_844_ai"
            self._deps_dir = Path(requirements_file_directory) / "Fusion"
            self._converter_dir = Path(converters_file_directory) / "FusionConverter"
            self._converter_venv_dir = Path(self._converter_dir) / "venv"
            self._builder = venv.EnvBuilder(with_pip=True) #for venv


        def _checkAlreadyInstalled(self) -> (bool, bool):
            """
            Checks if dependecies are already installed for the Fusion Platform, inspectionating the .installed.json file. 

            Parameters
            -----------
            - self
                
            Returns
            -------
            - install_needed: bool
            Boolean variable to install the needed dependencies. (These Dependencies are specified in self._deps_dir)
            - install_gpu: bool (For Future Exetensions)

            """

            requirementInstalled = {}
            try:
                with open(requirements_installed_path, "r") as installed_requirements:
                    requirementInstalled = load(installed_requirements)

                installed = requirementInstalled.get(self._platform, False)

                return installed, False

            except decoder.JSONDecodeError as e:
                logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
                logger.info(f"Installing only basic dependencies...")
                return True, False

            except (FileNotFoundError,Exception) as e:
                logger.critical(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            
            exit(0)


        def _downloadDependencies(self, platform: str, installed_requirements_dict: Dict[str, bool]) -> None:
            """
            Downloads the dependencies for Fusion platform. The ones stored in Fusion/fusion_converter.txt
            are specific for Converters/FusionConverter virtual environment.

            Parameters
            ----------
            - platform: str
            The specific target platform.

            - installed_requirements_dict: Dict
            The .installed.json dict object.

            Returns
            -------
            - None

            """
            
            requirements_file_fusion_path = str(self._deps_dir / "fusion.txt")
            requirements_file_fusion_converter_dependencies_path = str(self._deps_dir / "fusion_converter.txt")

            try:
                logger.info(f"INSTALLING {platform.upper()} BASIC DEPENDENCIES...")
                sleep(1)
                return_value_basic = check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_fusion_path])

                logger.info(f"INSTALLED BASIC DEPENDENCIES! PASSING TO {platform.upper()} CONVERTER ONES...")

                self._builder.create(self._converter_venv_dir)

                logger.info(f"VIRTUAL ENV FOR {self._platform} CONVERTER CREATED. INSTALLING DEPENDENCIES...\n")

                return_value_converter = check_call([os.path.join(self._converter_venv_dir, "bin", "python3.10"),"-m", "pip", "install", "-r", requirements_file_fusion_converter_dependencies_path])

                logger.info("INSTALLED DEPENDENCIES IN VENV!")

                if return_value_basic == 0 and return_value_converter == 0:
                    installed_requirements_dict[platform] = True


            except CalledProcessError as e:
                logger.critical(f"Encountered error installing dependencies.\nThe specific error is {e}.")
                exit(1)





if __name__ == "__main__":
    there_is_gpu = False

    pdm = PackageDownloadManagerFusion()

    pdm.checkDownloadedDependencies(there_is_gpu)
    