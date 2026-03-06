from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


from psutil import  cpu_count, cpu_percent, virtual_memory, disk_partitions, disk_usage
from GPUtil import getGPUs
#from amdsmi import init_amd_smi_lib, get_gpu_device_handles
from pyamdgpuinfo import detect_gpus, get_gpu
from platform import uname
from rich.pretty import pprint
from Utils.utilsFunctions import getHumanReadableValue, initialPrint

#Tests
#from PackageDownloadModule.packageDownloadManager import PackageDownloadManager
#from ConfigurationModule.configurationManager import ConfigManager

#TODO: Take some measurements in order to set the REAL thresholds.
default_hardware_empty_message="N.A."
default_memory_total_threshold=4294967296 #Total Memory Required: 4Gb 
default_memory_usage_threshold=2684354560 #Free Memory Required: 2.5Gb
default_disk_usage_threshold=2147483648 #Free Disk Required: 2Gb 
default_disk_total_threshold=6442450944 #Total Disk Required: 4Gb
default_cpu_usage_threshold=70 #CPU Usage Threshold
interval_cpu_usage=2 #interval for CPU Usage Check
VALID_CHOICES = {'y','n'} #Choices for CPU Usage


class ProbeHardwareManager():

    def __init__(self):
        self.__uname = uname()
        
    def __printInformations(self, input: dict, topic: str) -> None:
        """
        Handler function to print System Informations and Usage on terminal.

        Input:
            - input: dict that contains couples key, value to print.
            - topic: the topic to print at the first line
        Output:
            - None

        """
        print("\n" +"-"*10 + '\x1b[32m' + topic + '\033[0m' + "-"*10)
        for key, value in input.items():
            if isinstance(value, dict):
                print(f"{key} ")
                for value_keys, value_number in value.items():
                    print(f"\t{value_keys}: {value_number if value_number else default_hardware_empty_message}")
                continue
            print(f"{key}: {value if value else default_hardware_empty_message}")
        print("-"*10 + "-"*len(topic)+"-"*10+"\n")


    def __retrieveSysInfo(self) -> str:
        """
        Retrieves system informations and shows it on terminal.

        """

        uname = self.__uname
        
        sysinfo = {
            "System": uname.system if uname.system else default_hardware_empty_message,
            "Node name": uname.node if uname.node else default_hardware_empty_message,
            "Release": uname.release if uname.release else default_hardware_empty_message,
            "Version": uname.version if uname.version else default_hardware_empty_message,
            "Machine/Processor": uname.machine if uname.machine else default_hardware_empty_message,
        }

        self.__printInformations(sysinfo, " SYSTEM INFORMATION ")

        arch_info = ""

        if sysinfo["Machine/Processor"].startswith("x86"):
            arch_info="x86"
        elif sysinfo["Machine/Processor"].startswith("aarch"):
            arch_info="aarch"
        else:
            arch_info = ""


        return arch_info



    def __retrieveCpuUsage(self) -> None:
        """
        Retrieves CPU Usage informations and shows it on terminal. If the CPU Usage for the given interval (default 2s)
        is Greater-Equal to default_cpu_usage_threshold it'll show a warning prompt in order to continue or stop the execution 
        of the tool. 

        """
        
        
        physical_cpus = cpu_count(logical=False)
        cpu_usage = cpu_percent(percpu=False, interval=interval_cpu_usage)

        cpuinfo = {
            "Physical CPUs": physical_cpus if physical_cpus else default_hardware_empty_message,
            "CPU Usage (%)": f"{cpu_usage}%"
        }

        self.__printInformations(cpuinfo, " CPU INFORMATION ")

        if cpu_usage >= default_cpu_usage_threshold:
            while True:
                choice = input(f"{cpu_usage} of CPU Usage detected. Do you want to continue? (y/n): ").lower()

                if choice in VALID_CHOICES:
                    if choice == 'y':
                        break
                    else:
                        logger.info("\nEXITING...")
                        exit(0)
                else:
                    print("Invalid Input. Please enter 'y' or 'n'.")
            print("\n")

                
    def __retrieveMemoryUsage(self) -> None:
        """
        Retrieves Memory Usage informations and shows it on terminal. If the Memory Usage is Lower-Equal to 
        defaultMemoryXThreshold (on Total or Free evaluation), it'll stop the execution.

        """

        vmem = virtual_memory()
        mem_infos = [vmem.total, vmem.available, vmem.used]
        results = [getHumanReadableValue(value) for value in mem_infos]

        memoryinfo = {
            "Total Memory" : results[0],
            "Memory Available": results[1],
            "Memory Used" : results[2],
            "Percent ": vmem.percent
        }

        self.__printInformations(memoryinfo, " MEMORY USAGE INFORMATION ")
        

        if mem_infos[0] <= default_memory_total_threshold:
            logger.critical(f"The tools requires at least {getHumanReadableValue(default_memory_total_threshold)} of Total Memory to run smoothly.")
            logger.info("EXITING...")
            exit(0)

        if mem_infos[1] <= default_memory_usage_threshold:
            logger.critical(f"The tools requires at least {getHumanReadableValue(default_memory_usage_threshold)} of Free Memory to run smoothly.")
            logger.info("EXITING...")
            exit(0)

    


    def __retrieveDiskUsage(self) -> None:
        """
        Retrieves Disk Usage informations and shows it on terminal. If the Disk Total/Usage is Lower-Equal to 
        defaultDiskXThreshold (on Total or Free evaluation), it'll stop the execution.

        """
        
        partitions = disk_partitions()

        partitions_info = {}
        for partition in partitions:
            if "home" in partition.mountpoint or "/" in partition.mountpoint:
                partitions_info["Mountpoint"] = partition.mountpoint
                partitions_info["Total Disk Space"] = getHumanReadableValue(disk_usage(partition.mountpoint).total)
                partitions_info["Free Disk Space"] = getHumanReadableValue(disk_usage(partition.mountpoint).free)
                break

                # if "home" in partition.mountpoint:
                #     break
    
        self.__printInformations(partitions_info, " DISK USAGE INFORMATION ")

        if disk_usage(partition.mountpoint).total <= default_disk_total_threshold:
            logger.critical(f"The tools requires at least {getHumanReadableValue(default_disk_total_threshold)} of Total Disk.")
            logger.info("EXITING...")
            exit(0)

        if  disk_usage(partition.mountpoint).free <= default_disk_usage_threshold:
            logger.critical(f"The tools requires at least {getHumanReadableValue(default_disk_usage_threshold)} of Free Disk.")
            logger.info("EXITING...")
            exit(0)


    def __checkAMDGpuAvailability(self, gpu_info: dict, there_is_gpu: bool, gpu_type: str) -> (bool,str):
        """
        Checks if there are AMD GPUs in the system. WORKS ONLY ON LINUX!

        Input:
            - gpu_info: dictionary
            - there_is_gpu: bool value
            - gpu_type: str value

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """

        #TODO: ROCM PROBE SUPPORT

        #AMD (not ROCM)
        try:
            amd_gpus = detect_gpus()
            if amd_gpus>0:
                logger.info(f"AT LEAST ONE AMD GPU FOUND.\n")
                for i in range(amd_gpus):
                    gpu = get_gpu(i)
                    vram_usage=getHumanReadableValue(gpu.query_vram_usage())
                    gpu_load = gpu.query_load()*100


                    gpu_info[f"AMD GPU {i} Name"] = gpu.name if gpu.name else f"GPU AMD {i}"
                    gpu_info[f"VRAM Usage AMD {i}"] = vram_usage if vram_usage else default_hardware_empty_message
                    gpu_info[f"GPU Load AMD {i}"] =  f"{str(gpu_load)}%" if gpu_load>=0 else default_hardware_empty_message
                    there_is_gpu = True
                    gpu_type="AMD"
        except (ValueError,Exception) as e:
            logger.warning(f"Encountered an error recognising AMD GPU/s. Maybe there aren't AMD device/s or you have some missing dependencies.\nThe specific error is: {e}.")


        return there_is_gpu, gpu_type

    def __checkNVIDIAGpuAvailability(self, gpu_info: dict, there_is_gpu: bool, gpu_type: str) -> (bool,str):
        """
        Checks if there are available NVIDIA GPUs in the system. Requires CUDA drivers.

        Input:
            - gpu_info: dictionary
            - there_is_gpu: bool value
            - gpu_type: str value

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """

        #NVIDIA
        try:
            gpus = getGPUs()

            if len(gpus)>0:
                logger.info(f"AT LEAST ONE NVIDIA GPU FOUND.\n")
                for i, gpu in enumerate(gpus):
                    gpu_info[f"NVIDIA GPU {i} Name"] = gpu.name if gpu.name else "NVIDIA GPU {i}"
                    gpu_info[f"VRAM Usage NVIDIA {i}"] = gpu.memoryUsed if gpu.memoryUsed else default_hardware_empty_message
                    gpu_info[f"GPU Load NVIDIA {i}"]= gpu.load*100 if gpu.load>=0 else default_hardware_empty_message

                there_is_gpu = True
                gpu_type="NVIDIA"
                
        except (ValueError,Exception) as e:
            logger.warning(f"Encountered an error recognising NVIDIA GPU/s. Maybe there aren't NVIDA device/s or you have some missing dependencies.\nThe specific error is: {e}.\n")

        return there_is_gpu, gpu_type

    def __retrieveGpuInfo(self) -> (bool, str):
        """
        Checks for GPU infos, calling the check dedicated functions.

        Input:
            None

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """


        gpu_info = {}
        there_is_gpu = False
        gpu_type = None
        
        there_is_gpu, gpu_type = self.__checkNVIDIAGpuAvailability(gpu_info, there_is_gpu, gpu_type)
        there_is_gpu, gpu_type = self.__checkAMDGpuAvailability(gpu_info, there_is_gpu, gpu_type)

        if gpu_info:
            self.__printInformations(gpu_info, " GPU INFORMATION ")

        return there_is_gpu, gpu_type


    def checkSystem(self) -> (bool, str, str):
        """
        Checks the system characteristics, thanks to utility functions, in order to see if the target device has a
        sufficient amount of resources to execute the tool.

        Input:
            - None
        Output:
            - there_is_gpu: bool
            - gpu_type: str, describes the GPU type/vendor.
            - sys_arch: str, describes system architecture, 'x86' or 'aarch'
        """
        initialPrint("HARDWARE PROBE")
        sys_arch = self.__retrieveSysInfo()
        self.__retrieveCpuUsage()
        self.__retrieveMemoryUsage()
        self.__retrieveDiskUsage()
        there_is_gpu, gpu_type = self.__retrieveGpuInfo()


        if not there_is_gpu:
            logger.info("GPU INSTANCES NOT FOUND.")
        else:
            logger.info(f"GPU {gpu_type} FOUND! YOU CAN MAY HAVE SOME TROUBLES WITH THE EXECUTION. MAKE SURE THAT ALL THE DEPENDENCIES FOR GPU INFERENCING ARE INSTALLED AND SETTED.")

        return there_is_gpu, gpu_type, sys_arch



# if __name__=="__main__":
#     logger.info("PROBING HARDWARE RESOURCES...\n")
#     probe = ProbeHardwareManager()

#     there_is_gpu, gpu_type, sys_arch = probe.checkSystem()


#     pdm = PackageDownloadManager()

#     pdm.checkDownloadedDependencies(there_is_gpu)

#     cm = ConfigManager(sys_arch, there_is_gpu)


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
#                 "device": "gpu",
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
#             }
#         },
#         "dataset": {
#             "data_dir": "./ModelData/Dataset/dataset_name",
#             "batch_size": 32
#         }
#     }

#     hash_val = cm.createConfigFile(configTest)










