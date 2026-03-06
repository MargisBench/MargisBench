from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import os
import sys
import cv2
import getpass
import numpy as np
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from Utils.utilsFunctions import createPathDirectory, getModelTransforms
from Utils.loadingSpinner import LoadingSpinner
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent



class Initializers(ABC):
    """
    Abstract Base Class defining the contract for platform initialization strategies.
    Enforces setup, conversion, and initialization steps.
    """

    @abstractmethod
    def setUpPlatform():
        """
        Abstract method to configure the physical platform (SSH, drivers, permissions).
        """
        pass

    @abstractmethod
    def convertModels():
        """
        Abstract method to convert/compile models into the target-specific format (e.g., .tflite, .hef).
        """
        pass

    @abstractmethod
    def initialize():
        """
        Abstract method that acts as the main entry point to run the full initialization pipeline.
        """
        pass

class GenericInitializer(Initializers):
    """
    Concrete initializer for Generic platforms (standard CPU/GPU).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_id: Optional[str]=None):
        """
        Constructor for GenericInitializer.

        Parameters
        ----------
        - config: dict, optional
          The global configuration dictionary.
        - config_id: str, optional
          The unique identifier for the current experiment.
        """

        self.__config = config
        self.__config_id = config_id

    def __setUpGenericBoard(self) -> None:
        """
        Internal function to initialize the generic platform. 
        It specifically checks and grants permissions for cache clearing (via /usr/bin/tee) 
        to ensure accurate benchmarking without sudo prompts during runtime.
        """
        logger.info("Searching if tee has root grant in sudoers file. May ask for password...")
        user = os.popen('whoami').read().strip()
        sudoers_line = f"{user} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"

        try:

            check_cmd = f"sudo grep -q '^{sudoers_line}' /etc/sudoers"
            result = subprocess.run(check_cmd, shell=True)


            if result.returncode==0:
                logger.info("THE SUDO GRANT FOR /usr/bin/tee IS ALREADY ALLOWED IN SUDOERS FILE!")
            else:
                logger.info("GRANT NOT FOUND. ADDING TO /ETC/SUDOERS.") 
                append_cmd = f"echo '{sudoers_line}' | sudo EDITOR='tee -a' visudo"
                subprocess.run(append_cmd, shell=True, check=True)

                #if the subprocess succeeds in the task
                logger.info(f"SUDOERS FILE UPDATED!")



        except subprocess.CalledProcessError as e:
            logger.error(f"A generic error occured during the Generic platform Initialization. The specific error is: {e}")
            logger.warning(f"The cleaning of the caches may require password for each cleaning.")

        # --- CPU frequency ---
        logger.info(f"Trying to Set up the CPU FREQUENCY")
        subprocess.run( 
            "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
            shell=True
        )

    def convertModels(self) -> None:
        """
        Implementation of model conversion. 
        For Generic platforms, ONNX is usually the final state, so this is mostly a placeholder.
        """
        logger.info(f"THE .ONNX FILE WILL BE ALWAYS GENERATED.")
    

    def setUpPlatform(self) -> None:
        """
        Implementation of platform setup. Calls the internal generic board setup.
        """
        self.__setUpGenericBoard()

    def initialize(self) -> None:
        """
        Main entry point: Converts models (if needed) and sets up the board.
        """
        self.convertModels()
        self.setUpPlatform()

    def getConfig(self) -> Dict[str, Any]:
        """
        Getter for the configuration dictionary.
        """
        return self.__config
    
    def getConfigID(self) -> str:
        """
        Getter for the configuration ID.
        """
        return self.__config_id

    def setConfig(self, config) -> None:
        """
        Setter for the configuration dictionary.
        """
        self.__config = config

    def setConfigID(self, config_id):
        """
        Setter for the configuration ID.
        """
        self.__config_id = config_id


class CoralInitializer(Initializers):
    """
    Concrete initializer for Google Coral (Edge TPU) platforms.
    Handles quantization, compilation to TFLite, and device communication via MDT.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None, config_id: Optional[str]=None):
        """
        Constructor for CoralInitializer.

        Parameters
        ----------
        - config: dict, optional
          The global configuration dictionary.
        - config_id: str, optional
          The unique identifier for the current experiment.
        """
        self.__config = config
        self.__config_id = config_id


    def __createCalibrationData(self) -> None:
        """
        Internal function that generates representative datasets for quantization.
        It calls an external script (`calibration_data.py`) to create numpy arrays used 
        during the ONNX to TFLite conversion process for full integer quantization.
        """
        calibration_script_path = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "calibration_data.py")
        dataset_path = str(PROJECT_ROOT/ self.__config["dataset"]["data_dir"])

        args_sets = []
        for model_config in self.getConfig()["models"]:
            arg_set = []
            weights = model_config['weights_class']
            weights = weights.split(".")[0]

            arg_set.append(model_config['model_name'])
            arg_set.append(weights)

            # Batch size and Image size
            arg_set.append(str(self.getConfig()['dataset']['batch_size']))
            arg_set.append(str(model_config['image_size']))

            arg_set.append(dataset_path)
            args_sets.append(arg_set)

        for args in args_sets:
            logger.info(f"CREATING CALIBRATION DATA WITH ARGS: {args}")
            subprocess.run([sys.executable, calibration_script_path] + args)

    def __createCoralModels(self) -> None:
        """
        Internal function that converts ONNX models to Quantized TFLite models.
        It utilizes the `onnx2tf` tool within a virtual environment.
        """
        # Setup venv paths
        venv_root = PROJECT_ROOT / "Converters" / "CoralConverter" / "venv"
        venv_bin = venv_root / "bin"
        onnx2tf_path = str(venv_bin / "onnx2tf")

        # Isolated environment for the subprocess
        sub_env = os.environ.copy()
        sub_env['PATH'] = str(venv_bin) + os.pathsep + sub_env.get("PATH", "")


        config_id = self.getConfigID()
        onnx_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / config_id
        tf_models_root = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / config_id


        #DUMMY FILE FOR ONNX2TF CONVERSION
        dummy_file = os.path.join(os.getcwd(), "calibration_image_sample_data_20x128x128x3_float32.npy")

        if not os.path.exists(dummy_file):
            print(f"Generating dummy {dummy_file} in {os.getcwd()} to bypass download...")
            dummy_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
            np.save(dummy_file, dummy_data)

                

        for model_config in self.getConfig()["models"]:
            model_name = model_config['model_name']
            
            # Base Model Conversion
            base_onnx = str(onnx_path / f"{model_name}.onnx")
            base_out_dir = tf_models_root / model_name / f"{model_name}Q"
            base_tflite = base_out_dir / f"{model_name}_full_integer_quant.tflite"
            base_calib = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "Calibration" / "CalibrationArrays" / f"{model_name}_calibration_data.npy")

            createPathDirectory(base_out_dir)

            if not base_tflite.exists():
                with LoadingSpinner(message="Converting of onnx model into tflite model"):
                    try:

                        base_cmd = [
                            onnx2tf_path, "-i", base_onnx, "-o", str(base_out_dir),
                            "-oiqt", "-iqd", "float32", "-oqd", "float32", "-qt", "per-channel", 
                            "-b", "1", "-dgc", "-cind", "input", base_calib, "[[[[0.0]]]]", "[[[[1.0]]]]"
                        ]
                        subprocess.run(base_cmd, env=sub_env, capture_output=True, text=True)

                    except subprocess.CalledProcessError as e:
                        logger.error(f"Subprocess failed: {'  '.join(base_cmd)}")
                        logger.error("--- STDERR ---")
                        logger.error(e.stderr)
                        raise e
            else: 
                logger.info(f"Base Tflite Model already exists at {base_tflite}")


            # Optimized Models Conversion 
            opts = self.getConfig().get("optimizations", {})
            for opti_key in opts.keys():
                opti_onnx = None
                
                if opti_key == "Pruning":
                    suffix = "pruned"
                elif opti_key == "Distillation":
                    suffix = "distilled"
                else:
                    continue # Skip Quantization 

                opti_onnx = str(onnx_path / f"{model_name}_{suffix}.onnx")
                opti_out_dir = tf_models_root / model_name / f"{model_name}Q_{suffix.capitalize()}"
                opti_tflite = opti_out_dir / f"{model_name}_{suffix}_full_integer_quant.tflite"

                if os.path.exists(opti_onnx):
                    if opti_tflite.exists():
                        logger.info(f"Optimized tflite model already exists at: {opti_tflite}")
                    else:
                        createPathDirectory(opti_out_dir)

                        logger.info(f"CONVERTING {opti_key} MODEL: {model_name}")
                        with LoadingSpinner(message="Converting of onnx model into tflite model"):
                            try:
                                opti_cmd = [
                                    onnx2tf_path, "-i", opti_onnx, "-o", str(opti_out_dir),
                                    "-oiqt", "-iqd", "float32", "-oqd", "float32", "-qt", "per-channel", 
                                    "-b", "1", "-dgc", "-cind", "input", base_calib, "[[[[0.0]]]]", "[[[[1.0]]]]"
                                ]
                                

                                subprocess.run(opti_cmd, env=sub_env, capture_output=True, text=True, check=True)
                                
                            except subprocess.CalledProcessError as e:

                                logger.error(f"Subprocess failed: {' '.join(opti_cmd)}")
                                logger.error("--- STDERR ---")
                                logger.error(e.stderr)
                                raise e
                else:
                    logger.warning(f"Expected optimized model {opti_onnx} not found. Skipping.")


        os.remove(dummy_file)

    def __compileCoralModelsForEdgeTPU(self) -> None:
        """
        Internal function that runs the `edgetpu_compiler` to translate standard TFLite models
        into EdgeTPU-compatible TFLite models.
        """
        compiler_bin = str(PROJECT_ROOT / "Converters" / "CoralConverter" / "coral_compiler" / "x86_64" / "edgetpu_compiler")
        
        config_id = self.getConfigID()
        tflite_dir = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / config_id
        
        for model_config in self.getConfig()["models"]:
            model_name = model_config['model_name']
            
            model_dir = tflite_dir / f"{model_name}"
            base_out_dir =  tflite_dir / f"{model_name}EdgeTPU"
            
            base_tflite_path = model_dir / f"{model_name}Q" /  f"{model_name}_full_integer_quant.tflite"
            base_out_path = base_out_dir / f"{model_name}_edgeTPU.tflite"

            if base_tflite_path.exists():
                createPathDirectory(base_out_dir)
                logger.info(f"COMPILING BASE EdgeTPU MODEL FOR {model_name}...")
                # edgetpu_compiler <file> -o <output_dir>
                subprocess.run([compiler_bin, str(base_tflite_path), "-o", str(base_out_dir)], capture_output=True, check=True)
                logger.info(f"MODEL SUCCESSFULLY COMPILED FOR EdgeTPU")

            opts = self.getConfig().get("optimizations", {})
            for opti_key in opts.keys():
                suffix = None
                if opti_key == "Pruning":
                    suffix = "Pruned"
                elif opti_key == "Distillation":
                    suffix = "Distilled"
                else:
                    continue

                # Path to the .tflite we created in createCoralModels
                opti_tflite_path = model_dir / f"{model_name}Q_{suffix}" / f"{model_name}_{suffix.lower()}_full_integer_quant.tflite"
                
                # Destination directory for the compiled version
                opti_out_dir = base_out_dir

                if opti_tflite_path.exists():
                    createPathDirectory(opti_out_dir)
                    logger.info(f"COMPILING {opti_key} EdgeTPU MODEL FOR {model_name}...")
                    subprocess.run([compiler_bin, str(opti_tflite_path), "-o", str(opti_out_dir)], capture_output=True, check=True)
                    logger.info(f"MODEL SUCCESFULLY COMPILED FOR EdgeTPU")
                else:
                    logger.warning(f"TFLite source for {opti_key} not found at {opti_tflite_path}. Skipping compilation.")  

    def __setUpCoralBoard(self) -> None:
        """
        Internal function that interacts with the connected Coral device using `mdt` (Mendel Development Tool).
        It pushes the compiled models, the dataset, and the execution scripts to the device.
        """

        # --- CPU and EdgeTPU freq + Fan ---

        subprocess.run(["mdt", "exec", "echo 50000 | sudo tee /sys/class/thermal/thermal_zone0/trip_point_4_temp"])

        subprocess.run([
            "mdt", 
            "exec", 
            "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        ])


        # -- Models Transfer --
        subprocess.run(["mdt", "exec", "mkdir", "/home/mendel/Models"], capture_output=True)
        config_id = self.getConfigID()
        tfdir = PROJECT_ROOT / "Converters" / "CoralConverter" / "TfModels" / config_id

        if not tfdir.exists():
            logger.error(f"Directory {tfdir} does not exist. Cannot push models.")
            return     

        edgetpu_dirs = [
            path for path in tfdir.iterdir()
            if path.is_dir() and path.name.endswith("EdgeTPU")
        ]

        tflite_paths = []
        for d in edgetpu_dirs:
            found_files = list(d.glob("*.tflite"))
            tflite_paths.extend(found_files)

        logger.debug(f"Found {len(tflite_paths)} models to push: {tflite_paths}\n")
        for file_path in tflite_paths:
            logger.info(f"Pushing {file_path.name}...")
            
            try:
                
                # Using mdt command
                subprocess.run(
                    ["mdt", "push", str(file_path), "/home/mendel/Models"], 
                    check=True, # Raises an error if the command fails
                    capture_output=True
                )
                logger.info(f"SUCCESSFULLY PUSHED MODEL {str(file_path)} TO GOOGLE CORAL: /home/mendel/Models")
                
            except subprocess.CalledProcessError:
                logger.error(f"FAILED to push {file_path.name}. Is the device connected?")
            except FileNotFoundError:
                logger.error("Error: 'mdt' command not found. Is it installed and in your PATH?")

        # -- Dataset Transfer --
        datadir = self.getConfig()['dataset']['data_dir'] + "/test"
        
        try:
            subprocess.run(["mdt", "exec", "rm", "-rf",  "/home/mendel/test"], capture_output=True)
            subprocess.run(
                    ["mdt", "push", str(datadir), "/home/mendel/"], 
                    check=True, # Raises an error if the command fails
                    capture_output=True
                )
        except subprocess.CalledProcessError:
                logger.error(f"FAILED to push {datadir}. Is the device connected?")

        # -- Scripts Transfer --
        coralscriptdir = PROJECT_ROOT / "PlatformInitializers" / "CoralScripts"
        #tracertool_path = "benchmark_model_arm8_final"
        #subprocess.run(["chmod", "+x", tracertool_path])
        subprocess.run(["mdt", "exec", "chmod", "+x", "benchmark_model_arm8_final"])
        
        for script in coralscriptdir.iterdir():
            if script.is_file():
                try:
                    subprocess.run(
                        ["mdt", "push", str(script)], 
                        check=True,
                        capture_output=True
                    )
                    logger.info(f"PUSHED SUCCESSFULLY THE FOLLOWING ITEM: {script.name}")
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to push {script.name}")

            
    def convertModels(self) -> None:
        """
        Main conversion pipeline for Coral: Calibration -> TFLite Conversion -> EdgeTPU Compilation.
        """
        self.__createCalibrationData()
        self.__createCoralModels()
        self.__compileCoralModelsForEdgeTPU()

    def setUpPlatform(self) -> None:
        """
        Sets up the Coral physical platform by pushing all necessary assets.
        """
        self.__setUpCoralBoard()

    def initialize(self) -> None:
        """
        Entry point to run the full Coral initialization process.
        """
        self.convertModels()
        self.setUpPlatform()

    def getConfig(self) -> Dict[str, Any]:
        """
        Getter for the configuration dictionary.
        """
        return self.__config
    
    def getConfigID(self) -> str:
        """
        Getter for the configuration ID.
        """
        return self.__config_id

    def setConfig(self, config: Dict[str, Any]) -> None:
        """
        Setter for the configuration dictionary.
        """
        self.__config = config

    def setConfigID(self, config_id: str) -> None:
        """
        Setter for the configuration ID.
        """
        self.__config_id = config_id


class FusionInitializer(Initializers):

    """
    Concrete initializer for Hailo-8 (Fusion 844) platforms.
    Handles Dataflow Compiler execution via Docker, dataset conversion, and SSH setup.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_id: Optional[str] = None):
        """
        Constructor for FusionInitializer.

        Parameters
        ----------
        - config: dict, optional
          The global configuration dictionary.
        - config_id: str, optional
          The unique identifier for the current experiment.
        """

        self.__config = config
        self.__config_id = config_id


    def __createCalibrationData(self) -> None:
        """
        Internal function that calls the script in order to create the Calibration Data.
        These are necessary for the Hailo Dataflow Compiler to quantize the models correctly.
        """

        calibration_script_path = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "Calibration" / "calibration_data.py")
        dataset_path = str(PROJECT_ROOT / self.__config["dataset"]["data_dir"])

        args_sets = []
        for model_config in self.getConfig()["models"]:
            arg_set = []
            weights = model_config['weights_class']
            weights = weights.split(".")[0]

            arg_set.append(model_config['model_name'])
            arg_set.append(weights)

            # Batch size and Image size
            arg_set.append(str(self.getConfig()['dataset']['batch_size']))
            arg_set.append(str(model_config['image_size']))

            arg_set.append(dataset_path)
            args_sets.append(arg_set)

        for args in args_sets:
            logger.info(f"CREATING CALIBRATION DATA WITH ARGS: {args}")
            subprocess.run([sys.executable, calibration_script_path] + args)



    def __createFusionModels(self) -> None: 
        """
        Internal function that starts a Docker container to execute the Hailo compiler pipeline.
        It converts ONNX models to HAR (Hailo Archive) and then to HEF (Hailo Executable Format).        
        """

        hailo_container_bash_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "fusion_compiler" / "docker_hailo_compiler.sh"
        hailo_models_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}"
        hailo_har_files_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "har_files"
        hailo_har_quantized_files_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "har_files_quantized"
        hailo_hef_files_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "hef_files"
        hailo_profiling_files_path = "./" / PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "memory_profiling_files"

        if not hailo_models_path.exists():
            old_umask = os.umask(0)
            os.mkdir(hailo_models_path, mode=0o777)
            os.mkdir(hailo_har_files_path, mode=0o777)
            os.mkdir(hailo_har_quantized_files_path, mode=0o777)
            os.mkdir(hailo_hef_files_path, mode=0o777)
            os.mkdir(hailo_profiling_files_path, mode=0o777)
            os.umask(old_umask)

        try:

            logger.info(f"STARTING THE DOCKER CONTAINER FOR THE CONVERSION OF THE FILES...\n")
            logger.info(f"THE CONVERSION MAY TAKE A WHILE...\n")
            result = subprocess.run([str(hailo_container_bash_path), self.__config_id], check=True)


            if result.returncode == 0:
                logger.info(f"ALL FILES CREATED CORRECTLY!")
                #CLEANING
                os.remove(str(hailo_models_path / "compile.sh"))
                os.remove(str(hailo_models_path / "create_memory_profiling.sh"))
                os.remove(str(hailo_models_path / "model_modifications.py"))
                return

        except KeyboardInterrupt as e:
            subprocess.run(["docker", "stop", "hailo8_compiler", "-s", "9"], check=True)
            logger.info(f"[INFO] DOCKER STOPPED.")
            
        except ChildProcessError as e:
            logger.error(f"Child Process Error starting the child process to create the Fusion .hef files.\nThe specific error is: {e}")
            exit(1)

        except Exception as e:
            logger.critical(f"Encountered a generic problem starting the child process to create the Fusion .hef files.\nThe specific error is: {e}")
            exit(1)

        logger.error(f"The subprocess exited with code {result.returncode}. Exiting...")
        exit(1)

    def __createDatasetBinFiles(self) -> None:
        """
        Internal function that converts the input image dataset into raw binary files (.bin).
        This is often required for embedded C++ inference pipelines on the target board.
        """

        dataset_conversion_script_path = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "DatasetConverter" / "dataset_converter.py")


        try:
            logger.info("CREATING THE BIN FILES FROM DATASET/TEST DIR...")

            converted = []
            for model_info in self.__config["models"]:
                transforms = getModelTransforms(model_info).transforms[0]
                if transforms.crop_size[0] not in converted:
                    resize_size=transforms.resize_size[0]
                    crop_size=transforms.crop_size[0]
                    subprocess.run([sys.executable, dataset_conversion_script_path, self.__config["dataset"]["data_dir"], str(crop_size), str(resize_size)])
                    converted.append(crop_size)


            logger.info("CONVERSION SUCCEDED!")

        except subprocess.CalledProcessError as e:
            logger.critical(f"An error is occurred converting the dataset in .bin files.\nThe specific error is: {e}")
            exit(1)

    def __setupFusionSSH(self, destination_path_host: str) -> Path: 
        """
        Internal function that ensures password-less SSH access to the Fusion board.
        It generates a specific key pair for the platform if missing and copies it to the target.
        
        Parameters
        ----------
        - destination_path_host: str
          The IP address or hostname of the target Fusion board.

        Returns
        -------
        - private_key_path: Path
          The filesystem path to the generated/used private key.
        """
        user = getpass.getuser()
        pub_key_path = Path(f'/home/{user}/.ssh/') / f"{self.__config['platform']}.pub"
        private_key_path = Path(f'/home/{user}/.ssh/') / f"{self.__config['platform']}"

        try:

            logger.info(f"CHECKING SSH CREDENTIALS...")
            
            if not (pub_key_path.exists() and private_key_path.exists()):
                logger.info(f"CREATING THE KEY FILES...")
                subprocess.run(['ssh-keygen', '-t', 'rsa', '-f', private_key_path, '-N', '', '-q'], check=True)
            else:
                logger.info("THE SSH ARE ALREADY CREATED.")

            logger.info(f"PUSHING THE CREDENTIALS IN {self.__config['platform']} PLATFORM. MAY ASK FOR PASSWORD.")

            subprocess.run(['ssh-copy-id', '-i', private_key_path, destination_path_host], check=True)

            logger.info(f"KEY SUCCESSFULLY PUSHED INTO THE DESTINATION HOST.")


        except subprocess.CalledProcessError as e:
            logger.critical(f"Encountered an error in child process meanwhile the SSH setup.\nThe specific error is: {e}")
            exit(1)
        except Exception as e:
            logger.critical(f"Encountered a generic error meanwhile the SSH setup.\nThe specific error is: {e}")
            exit(1)

        return private_key_path



    def __setupFusionBoard(self, destination_path_host: str, destination_host: str, private_key_path: Path) -> None:
        """
        Internal function that pushes all required assets to the Fusion board via SCP.
        Assets include: dataset binaries, compiled HEF models, Hailo runtime libraries, and C inference scripts.

        Parameters
        ----------
        - destination_path_host: str
          SCP-style path (e.g., "user@ip:/path/to/dest").
        - destination_host: str
          SSH host address (e.g., "user@ip").
        - private_key_path: Path
          Path to the SSH private key for authentication.
        """
        dataset_path = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "DatasetConverter" / "DatasetFiles")
        hef_models_path = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "hef_files")
        profiling_mem_models_path = str(PROJECT_ROOT / "Converters" / "FusionConverter" / "HefModels" / f"{self.__config_id}" / "memory_profiling_files")
        lib_hailo_path = str(PROJECT_ROOT / "PlatformInitializers" / "FusionScripts" /"libhailort")
        infer_script_path = str(PROJECT_ROOT / "PlatformInitializers" / "FusionScripts" / "inference.c")
        run_suite_script_path = str(PROJECT_ROOT / "PlatformInitializers" / "FusionScripts" / "run_suite.sh")


        try:

            logger.warning("Due to the nature of the target, something can go wrong. In most cases these problems are related to incorrect configurations.")

            logger.info("CONNECTING TROUGH SSH TO THE FUSION TARGET. MAY ASK FOR ROOT PASSWORD ON THE DEVICE...")
            with LoadingSpinner(message="SENDING FILES"):
                subprocess.run(['scp', '-q', '-i', private_key_path, '-r', dataset_path, hef_models_path, profiling_mem_models_path, lib_hailo_path, infer_script_path, run_suite_script_path, destination_path_host], check=True)
            
            with LoadingSpinner(message="COMPILING THE INFERENCE SCRIPT"):
                subprocess.run(['ssh', '-q', '-i', private_key_path, '-t', destination_host, 'aarch64-ttcontrol-linux-gcc -o inference inference.c -lpthread /usr/lib/libhailort.so.4.20.1 -I./libhailort/include/hailo; chmod +x ./run_suite.sh'], check=True)

            with LoadingSpinner(message="SETTING PERFORMANCE MODE"):
                subprocess.run(['ssh', '-q', '-i', private_key_path, '-t', destination_host, 'echo "performance" |tee  /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'], check=True)

            logger.info("SETUP COMPLETE!")

        except subprocess.CalledProcessError as e:
            logger.critical(f"Encountered an error in child process tranfering the files to the target.\nThe specific error is: {e}")
            exit(1)
        except Exception as e:
            logger.critical(f"Encountered a generic error tranfering the files to the target.\nThe specific error is: {e}")
            exit(1)


    def initialize(self) -> None:
        """
        Entry point to run the full Fusion initialization process.
        """
        self.convertModels()
        self.__createDatasetBinFiles()
        self.setUpPlatform()

    def convertModels(self)-> None:
        """
        Main conversion pipeline for Fusion: Calibration -> HEF Compilation.
        """
        self.__createCalibrationData()
        self.__createFusionModels()

    def setUpPlatform(self) -> None:
        """
        Sets up the Fusion physical platform by configuring SSH and pushing compiled assets.
        Relies on 'FUSION_HOST_IP' environment variable.
        """
        destination_host=os.getenv("FUSION_HOST_IP")
        destination_path_host = destination_host + ':/home/root'
        ssh_key_path = self.__setupFusionSSH(destination_host)
        self.__setupFusionBoard(destination_path_host, destination_host, ssh_key_path)

    def getConfig(self) -> Dict[str, Any]:
        """
        Getter for the configuration dictionary.
        """
        return self.__config
    
    def getConfigID(self) -> str:
        """
        Getter for the configuration ID.
        """
        return self.__config_id

    def setConfig(self, config) -> None:
        """
        Setter for the configuration dictionary.
        """
        self.__config = config

    def setConfigID(self, config_id: str) -> None:
        """
        Setter for the configuration ID.
        """
        self.__config_id = config_id


if __name__ == "__main__":

    #config_id = "6bae1867a5_generic"

    config_id = "1ce0af586313_fusion"

    config = {
        "models": [
            {
                "module": "torchvision.models",
                "model_name": "mobilenet_v2",
                "native": False,
                "weights_path": "ModelData/Weights/mobilenet_v2.pth",
                "device": "cpu",
                "class_name": "mobilenet_v2",
                "weights_class": "MobileNet_V2_Weights.DEFAULT",
                "image_size": 224,
                "num_classes": 2,
                "task": "classification",
                "description": "Mobilenet V2 from torchvision"
            },
            {
                "module": "torchvision.models",
                "model_name": "efficientnet",
                "native": True,
                "weights_path": "./ModelData/Weights/casting_efficientnet_b0.pth",
                "device": "cpu",
                "class_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights.DEFAULT",
                "image_size": 224,
                "num_classes": 2,
                "task": "classification",
                "description": "EfficientNet from torchvision"
            },
            {
                "module": "torchvision.models",
                "model_name": "mnasnet1_0",
                "native": False,
                "weights_path": "ModelData/Weights/mnasnet1_0.pth",
                "device": "cpu",
                "class_name": "mnasnet1_0",
                "weights_class": "MNASNet1_0_Weights.DEFAULT",
                "image_size": 224,
                "num_classes": 2,
                "task": "classification",
                "description": "mnasnet_v2 from torchvision"
            }
        ],
        "optimizations": {
            "Pruning": {
                "method": "LnStructured",
                "n": 1,
                "amount": 0.1,
                "epochs": 1
            },
            "Distillation": {
                "method": True,
                "distilled_paths": {
                    "mobilenet_v2": "/home/frabru99/Sync/MasterThesis/MargisBench/ModelData/Weights/mobilenet_v2_distilled.pth",
                    "efficientnet": "/home/frabru99/Sync/MasterThesis/MargisBench/ModelData/Weights/efficientnet_b0_distilled.pth",
                    "mnasnet1_0": "/home/frabru99/Sync/MasterThesis/MargisBench/ModelData/Weights/mnasnet1_0_distilled.pth"
                }
            }
        },
        "dataset": {
            "data_dir": "ModelData/Dataset/casting_data",
            "batch_size": 1
        },
        "repetitions": 2,
        "platform": "fusion_844_ai",
        "arch": "x86_64"
    }


    #coral_init = CoralInizializer(config, config_id) 
    # coral_init.createCalibrationData()
    # coral_init.createCoralModels()
    # coral_init.compileCoralModelsForEdgeTPU()
   

    #fusion_init = FusionInitializer(config, config_id)
    #fusion_init.createCalibrationData()

    #coral_init.setUpPlatform()

    fusion_init = FusionInitializer(config, config_id)
    fusion_init.initialize()