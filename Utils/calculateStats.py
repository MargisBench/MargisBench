from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import pandas as pd
from os import remove
from json import JSONDecodeError, load
from abc import ABC, abstractmethod
from numpy import percentile
from Utils.utilsFunctions import getHumanReadableValue
from typing import Dict, List, Any, Optional, Tuple, Union



class CalculateStats(ABC):
    """
    Abstract Class to calculate the stats in different ways, tahnks to the strategy pattern. 
    
    Contains only a single function (CalculateStats) to override in order to specify the adopted strategy.

    """
    @abstractmethod
    def calculateStats():
        pass

class CalculateStatsGeneric(CalculateStats):

    def printStats(input: Dict[str, Union[str, float]], topic: str) -> None:
        """
        Handler function to print Stas of the model on terminal.

        Input:
        - input: Dict
        Dict that contains couples key, value to print.

        - topic: str
        The topic to print at the first line.

        Returns
        -------
        - None

        """
        print("\n" +"-"*10 + '\x1b[32m' + topic + '\033[0m' + "-"*10)
        for key, value in input.items():
            if key=="Accuracy":
                print("\n")
                
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
                continue
            
            print(f"{key}: {value}")
                

        print("-"*10 + "-"*len(topic)+"-"*10+"\n")

    def printConfusionMatrix(cm: Any, topic: str, class_names: List[str]) -> None:
        """
        Handler function to print Confusion Matrix of the model on terminal.

        Parameters
        ----------
        - cm: Any
        Object that contains the values to print,
        - topic: str
        The topic to print at the first line
        - class_names: List
        The list of class names

        Returns
        -------
        - None 
        
        """
        print("\n" +"-"*5 + '\x1b[32m' + topic + '\033[0m' + "-"*5)
        # Header
        header_label = "True \\ Pred"
        print(f"{header_label:<15}", end="")
        for c in class_names:
            print(f"{c[:10]:<12}", end="") 
        print()
        
        # Rows
        for i, row_name in enumerate(class_names):
            print(f"{row_name:<15}", end="")
            for val in cm[i]:
                print(f"{val:<12}", end="")
            print()

        print("-"*5 + "-"*len(topic)+"-"*5+"\n")


    def _calculateKernelStats(profile_file_path: str, num_batches: int, total_images: int, correct: int, total: int, running_loss: int) -> Dict[str, float]:
        """
        Parses an ONNX Runtime profile JSON file to get pure kernel statistics.
        
        Parameters
        ----------
            - profile_file_path : str  
                        The path to the profile.json file.
            - num_batches : int 
                        The total number of batches in the inference run (e.g., len(inference_loader)).
            - total_images : int 
                    The total number of images in the dataset (e.g., len(inference_loader.dataset)).
            - correct : int 
                    Total correct prediction counter
            - total : int 
                    Total prediction counter
            - running_loss : int 
                    Calculated loss.

        Returns
        -------
            - stats : Dict 
                    A dictionary with total time, per-batch avg, and per-image avg.

        """
        
        total_kernel_time_us = 0
        total_model_run_time_us = 0
        total_sequential_execution_time_us = 0
        max_layer_memory_arena_consumption= 0
        
        memory_weights_consumption = 0
        layer_finished  = False


        node_events = []
        

        try:
            with open(profile_file_path, 'r') as f:
                trace_data = load(f)

            # Iterate over all events in the trace
            for event in trace_data:
                event_cat = event.get("cat")
                event_name = event.get("name")
                event_dur = event.get("dur", 0)

                if event_cat  == "Node":
                    duration_us = event_dur
                    total_kernel_time_us += duration_us
                    node_events.append(duration_us)

                    if not layer_finished:
                        memory_arena_consumption = int(event.get("args").get("output", 0)) + int(event.get("args").get("activation_size", 0))
                        if memory_arena_consumption > max_layer_memory_arena_consumption:
                            max_layer_memory_arena_consumption = memory_arena_consumption

                    if not layer_finished:
                        memory_weights_consumption += int(event.get("args").get("parameter_size", 0))

                elif event_cat == "Session" and event_name == "model_run":
                    total_model_run_time_us += event_dur

                elif event_cat == "Session" and event_name == "SequentialExecutor::Execute":
                    total_sequential_execution_time_us += event_dur
                    layer_finished=True
            
            if num_batches == 0 or total_images == 0:
                logger.error(f"Number of batches or images cannot be zero. im: {total_images}; batch: {num_batches}")
                return {}

            if not node_events:
                logger.warning(f"No Node events found in {profile_file_path}.")
                return {}

            # Calculate the stats to return
            total_kernel_time_ms = total_kernel_time_us / 1000.0
            total_model_run_time_ms = total_model_run_time_us / 1000.0
            total_sequential_execution_time_ms = total_sequential_execution_time_us / 1000.0

            avg_kernel_time_per_batch_ms = total_kernel_time_ms / num_batches
            avg_kernel_time_per_image_ms = total_kernel_time_ms / total_images
            avg_sequential_executor_time_per_batch_ms = total_sequential_execution_time_ms / num_batches
            avg_sequential_executor_time_per_image_ms = total_sequential_execution_time_ms / total_images
            avg_model_run_time_per_batch_ms = total_model_run_time_ms / num_batches
            avg_model_run_time_per_image_ms = total_model_run_time_ms / total_images
            
            total_onnx_runtime_overhead = total_model_run_time_ms - total_sequential_execution_time_ms
            avg_onnx_runtime_overhead_per_batch_ms = avg_model_run_time_per_batch_ms - avg_sequential_executor_time_per_batch_ms
            avg_onnx_runtime_overhead_per_image_ms = avg_model_run_time_per_image_ms - avg_sequential_executor_time_per_image_ms

            fps = 1000 / avg_model_run_time_per_image_ms

            node_latencies_ms = [n / 1000.0 for n in node_events]
            p95_node_latency_ms = percentile(node_latencies_ms, 95)

            accuracy = 100 * correct / total
            average_loss = running_loss / total

            logger.debug(f"Inference Event Path: {profile_file_path}")

            stats = {
                "Total 'kernel' inference time": total_kernel_time_ms,
                "Avg. 'kernel' inference time per batch": avg_kernel_time_per_batch_ms,
                "Avg. 'kernel' inference time per image": avg_kernel_time_per_image_ms,
                "Total sequential executor time": total_sequential_execution_time_ms,
                "Avg. sequential executor time per batch": avg_sequential_executor_time_per_batch_ms,
                "Avg. sequential executor time per image": avg_sequential_executor_time_per_image_ms,
                "Total model run time": total_model_run_time_ms,
                "Avg. model run time per batch": avg_model_run_time_per_batch_ms,
                "Avg. model run time per image": avg_model_run_time_per_image_ms,
                "Total ONNX runtime overhead": total_onnx_runtime_overhead,
                "Avg. ONNX runtime overhead per batch": avg_onnx_runtime_overhead_per_batch_ms,
                "Avg. ONNX runtime overhead per image": avg_onnx_runtime_overhead_per_batch_ms,
                "FPS": fps,
                "total_nodes_executed": len(node_events),
                "p95_node_latency_ms": p95_node_latency_ms,
                "Max Memory Arena Consumption": getHumanReadableValue(max_layer_memory_arena_consumption), 
                "Weights Memory Consumptions": getHumanReadableValue(memory_weights_consumption),
                "Accuracy": accuracy,
                "Avg. Loss": average_loss
            }

        
            # Clean up the file
            try:
                remove(profile_file_path)
                logger.debug(f"Cleaned up profile file: {profile_file_path}")
            except OSError as e:
                logger.warning(f"Could not delete profile file {profile_file_path}: {e}")

            return stats

        except FileNotFoundError:
            logger.error(f"Profile file not found: {profile_file_path}")
            return {}
        except JSONDecodeError:
            logger.error(f"Error decoding JSON from {profile_file_path}")
            return {}
        except Exception as e:
            logger.error(f"An error occured during profiling: {e}")
            return {}


    def calculateStats(profile_file_path: str, num_batches: int, total_images: int, correct: int, total: int, running_loss: int) -> Dict[str, float]:
        """
        Calculates the Kernel Stats, Ram Usage for each model inferencing for Generic platform. 


        Parameters
        ----------
        - profile_file_path: str
                        The path to the profile.json file.
        - num_batches: int
                        The total number of batches in the inference run
                    (e.g., len(inference_loader)).
        - total_images: int
                        The total number of images in the dataset
                    (e.g., len(inference_loader.dataset)).
        - correct: int
                        Total correct prediction counter
        - total: int
                        Total prediction counter
        - running_loss: int
                        Calculated loss

        Returns
        -------
        - state_dict: Dict
                        A dictionary with total time, per-batch avg, and per-image avg.
        
        """

        state_dict = {}

        try:
            state_dict = CalculateStatsGeneric._calculateKernelStats(profile_file_path, num_batches, total_images, correct, total, running_loss)
            # TODO: RAM TRACING
        except Exception as e:
            logger.error(f"Encountered a generic error calculating kernel stats.\nThe specific error is: {e}")


        return state_dict

class CalculateStatsCoral(CalculateStats): 


    def calculateStats(csv_path: str, num_batches: int) -> Dict[str, float]:
        """
        Calculates the Kernel Stats, Ram Usage for each model inferencing for Coral platform. 


        Parameters
        ----------
        - csv_path : str
                        The specific path for the csv pulled from the device target
        - num_batches : int
                        The number of batches.


        Returns
        -------
        - state_dict: Dict
                    A dictionary with Accuracy, InferenceAvgTime, InitTime, PeakMemory
        """
        
        # Take the results and convert them in stats
        logger.debug(f"This is the path of the temp results for Coral: {csv_path}")
        logger.debug(f"There are {num_batches} batches")
        df = pd.read_csv(csv_path)
        logger.debug(f"HERE THE RESULTS FROM THE CORAL")
        logger.debug(df)

        accuracy = df['Accuracy(%)'].iloc[0]
        inference_avg = df['InferenceAvg(ms)'].iloc[0]
        init_time = df['InitTime(ms)'].iloc[0]
        peak_memory = df['PeakMemory(MB)'].iloc[0]

        total_model_run = inference_avg * num_batches
        fps = 1000 / inference_avg

        stats = {
            "Total model run time": total_model_run,
            "Avg. model run time per batch": inference_avg,
            "Init Time": init_time,
            "Peak Memory": peak_memory,
            "FPS": fps,
            "Accuracy": accuracy
        }

        logger.debug(f"These are the stats in the dictionary: {stats}")

        return stats


class CalculateStatsFusion(CalculateStats):

    def calculateStats(csv_path: str, num_batches: int):
        """
        Calculates the Kernel Stats, Ram Usage for each model inferencing for Fusion platform. 


        Parameters
        ----------
        - csv_path : str
                    The specific path for the csv pulled from the device target
        - num_batches : int
                    The number of batches.

        Returns
        -------
        - state_dict: dict
                    A dictionary with Accuracy, InferenceAvgTime, InitTime, PeakMemory, FPS and more.
        """
        
        logger.debug(f"This is the path of the temp results for Fusion: {csv_path}")
        logger.debug(f"There are {num_batches} batches")
        df = pd.read_csv(csv_path)
        logger.debug(f"HERE THE RESULTS FROM THE FUSION")
        logger.debug(df)

        accuracy = df['Accuracy(%)'].iloc[0]
        total_model_run = df['InferenceAvg(ms)'].iloc[0]
        avg_overall_per_batch = df['AvgOverallLatencyPerBatch(ms)'].iloc[0]
        avg_overall_per_image= df['AvgOverallLatencyPerImage(ms)'].iloc[0]
        avg_hw_per_batch = df['AvgHwLatencyPerBatch(ms)'].iloc[0]
        avg_hw_per_image = df['AvgHwLatencyPerImage(ms)'].iloc[0]
        peak_memory = df['PeakMemory(MB)'].iloc[0]
        peak_memory_accelerator = df['PeakMemoryAccelerator(MB)'].iloc[0]
        weights_memory = df['WeightsMemory(MB)'].iloc[0] / (1024 * 1024)
        

        overhead_per_batch= avg_overall_per_batch - avg_hw_per_batch 
        overhead_total = overhead_per_batch * num_batches
        total_kernel_run_time = avg_hw_per_batch * num_batches 

        fps = 1000 * num_batches / total_model_run

        stats = {
            "Total model run time": total_model_run, 
            "Avg. 'kernel' inference time per batch": avg_hw_per_batch,
            "Avg. 'kernel' inference time per image": avg_hw_per_image,
            "Total kernel run time": total_kernel_run_time,
            "Avg. model run time per batch": avg_overall_per_batch,
            "Avg. model run time per image": avg_overall_per_image,
            "Overhead per batch": overhead_per_batch,
            "Total Overhead": overhead_total,
            "FPS": fps,
            "Peak Memory": peak_memory,
            "Peak Memory Accelerator": peak_memory_accelerator,
            "Weights Memory": weights_memory,
            "Accuracy": accuracy, 
        }

        logger.debug(f"These are the stats in the dictionary: {stats}")

        return stats