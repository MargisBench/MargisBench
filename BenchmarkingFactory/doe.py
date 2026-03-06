from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


import os
import torch
import json
import math
import gc
import sys
import traceback
import pingouin as pingu
from pathlib import Path
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from copy import deepcopy
from scipy import stats
from statsmodels.formula.api import ols
from itertools import product
import matplotlib.pyplot as plt
import statsmodels.api as sm
from multiprocessing import Process, SimpleQueue, set_start_method, get_context 
from typing import Dict, List, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader

import BenchmarkingFactory.aiModel as aiModel
import BenchmarkingFactory.optimization as optimization
import BenchmarkingFactory.dataWrapper as datawrapper
from ConfigurationModule.configurationManager import optimizations_library_path, ConfigManager

# Local Imports for Type Hinting
from BenchmarkingFactory.aiModel import AIModel
from BenchmarkingFactory.optimization import Optimization
from BenchmarkingFactory.dataWrapper import DataWrapper
import BenchmarkingFactory.optimization as optimization_module # Alias to avoid name conflict with class
import BenchmarkingFactory.aiModel as aiModel_module
import BenchmarkingFactory.dataWrapper as datawrapper_module

from Utils.utilsFunctions import initialPrint, subRunQueue
from PlatformContext.platform_context import PlatformContext
from rich.pretty import pprint


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#For Example Purposes
#from ProbeHardwareModule.probeHardwareManager import ProbeHardwareManager
from PackageDownloadModule.packageDownloadManager import PackageDownloadManager


torch.multiprocessing.set_sharing_strategy('file_system')


class DoE():
    _instance = None #for Singleton

    #Singleton Management
    def __new__(cls, *args, **kwargs) -> object:
        """
        Singleton __new__ method to ensure only one instance of DoE exists.

        Parameters
        ----------
        - cls: The class itself

        Returns
        -------
        - instance: The singleton instance of the DoE class
        """
        if cls._instance is None:
            cls._instance=super().__new__(cls)
        return cls._instance

    ###################################################################

    # TODO
    # SOME DATA IN THE DOE CONFIG MUST BE ADDED TO THE GENERARL CONFIG

    ###################################################################


    def __init__(self, context: PlatformContext, config: Dict[str, Any], config_id: str) -> None:
        """
        Constructor to initialize the Design of Experiments (DoE) manager.

        Parameters
        ----------
        - context: PlatformContext
          The context object abstraction for the underlying hardware/platform
        - config: dict
          The configuration dictionary containing models, optimizations, dataset info, and repetitions
        - config_id: str
          The unique identifier for this specific experiment run
        """
        if not hasattr(self, "created"):
            self.created=True #After that the first object is initialized, we'll not initialize another one. 
            self.__initialized=False
            self.__ran=False
            self.__context = context
            self.__platform = config["platform"]
            self.__config_id = config_id
            self.__steps = config.get("optimizations", {}).get("Pruning", {}).get("epochs", 0)
            self.__repetitions = config["repetitions"] 
            self.__model_info=config["models"]
            self.__optimizations_info=config["optimizations"]
            self.__dataset_info = config["dataset"]
            self.__models=self.__initializeListOfModels(config["models"])
            self.__optimizations=self.__initializeListOfOptimizations(config["optimizations"])
            self.__dataset=self.__initializeDataset()
            self.__inference_loaders={}
            self.__design = self.__initializeDesign()
            self.__current_stats = None
        #metrics?

    def __initializeListOfModels(self, models: List[Dict[str, Any]]) -> Dict[str, Dict[str, AIModel]]:
        """ 
        Internal function that initializes the models list starting from the config.

        Parameters
        ----------
        - models: list
          List of dictionaries, each containing configuration for a specific AIModel

        Returns
        -------
        - ai_models_dict: dict
          Dictionary where keys are model names and values are dictionaries of AIModel objects
        """

        initialPrint("MODEL INITIALIZATION")
        ai_models_dict={}

        for aiModelDict in models:
            logger.info(f"CREATING THE {aiModelDict['model_name']} MODEL...\n")
            ai_models_dict[aiModelDict['model_name']] = {}   
            ai_models_dict[aiModelDict['model_name']]['Base'] = aiModel.AIModel(aiModelDict)

        logger.info(f"MODELS CREATED!\n")


        return ai_models_dict


    def __initializeListOfOptimizations(self, optimizations: Dict[str, Any]) -> List[Tuple[Optimization, str]]:
        """ 
        Internal function that initializes the optimization objects based on the config.

        Parameters
        ----------
        - optimizations: dict
          Dictionary containing configuration for various optimizations (Pruning, Quantization, etc.)

        Returns
        -------
        - optimization_object_list: list
          List of tuples containing (OptimizationObject, OptimizationName)
        """
        
        optimization_object_list=[]

        try:
        
            for optimization_name in optimizations.keys():
                full_class_name=f"{optimization_name}Optimization"
                target_class = getattr(optimization, full_class_name)

                optimization_object = target_class(optimizations[optimization_name])
                optimization_object_list.append((optimization_object, optimization_name))
                logger.info(f"{full_class_name} ADDED!")

        except (FileNotFoundError,Exception) as e:
            logger.error(f"Encountered a generic problem initializing the list of optimizations.\nThe specific error is: {e}.")
  

        return optimization_object_list
        


    def __initializeDataset(self) -> DataWrapper:
        """
        Internal function to initialize the DataWrapper object. 

        Returns
        -------
        - dataset_wrapper: DataWrapper
          The object responsible for handling data loading and processing
        """

        dataset_wrapper = datawrapper.DataWrapper()

        return dataset_wrapper

    def __initializeDesign(self) -> List[Tuple[str, str]]:
        """
        Internal function to create the Full Factorial Design of experiments.
        Generates all combinations of Models x Optimizations x Repetitions.

        Returns
        -------
        - full_design: list
          A list of tuples (ModelName, OptimizationName) representing the execution order
        """

        initialPrint("DESIGN OF EXPERIMENTS\n")
        models_list = []
        optimization_list = []

        for model_name in self.__models.keys():
            models_list.append(model_name)
            
        optimization_list.append("Base")
        for opt_obj, opt_name in self.__optimizations:
            optimization_list.append(opt_name)

        design = list(product(models_list, optimization_list))

        self.__printDesign(design)
        full_design = []
        for _ in range(self.__repetitions):
            full_design.extend(design)

        return full_design

    def __printDesign(self, design: List[Tuple[str, str]]) -> None:
        """
        Helper function to print the generated design table to the console.
        """
        print('\x1b[32m' +f"{'MODEL NAME':<20}\t{'OPTIMIZATION':<20}\tREPETITIONS" + '\x1b[37m')
        for model, optimization in design:
            print(f"{model:<20}\t{optimization:<20}\t{self.__repetitions}")

        print("\n")


    def getContext(self) -> PlatformContext:
        """
        Returns the platform context associated with this DoE instance.
        """
        return self.__context

    def initializeDoE(self) -> None:
        """
        Main initialization routine. 
        1. Loads the dataset.
        2. Applies configured optimizations (Pruning/Quantization/Distillation) to the base models.
        3. Generates ONNX files for inference.

        Parameters
        ----------
        None
        """

        initialPrint("OPTIMIZATIONS APPLY")
        optimized_models = []
        

        for model_key, model_dict in self.__models.items(): #Iterating the base models. 
            dataset = self.__dataset #creating the dedicated dataWrapper for the model
            dataset.loadInferenceData(model_info=model_dict['Base'].getAllInfo(), dataset_info=self.__dataset_info)
            inference_loader = dataset.getLoader()
            finetune_loader = dataset.getFineTuningLoader()
            self.__inference_loaders[model_dict['Base'].getInfo("model_name")]=inference_loader #N Base Models = N Loaders
            model_dict['Base'].createOnnxModel(inference_loader, self.__config_id)

            #Optimized Model...
            for optimizator, op_name in self.__optimizations:

                calibration_samples = 30

                calibration_loader = dataset.getCalibrationLoader(num_samples = calibration_samples)

                optimizator.setAIModel(model_dict['Base'])
                optimized_model, already_created = optimizator.applyOptimization(self.__steps, inference_loader, calibration_loader, finetune_loader, self.__config_id)

                if not optimized_model.getInfo("model_name").endswith("quantized") and not already_created:
                    optimized_model.createOnnxModel(inference_loader, self.__config_id)

                #optimized_models.append(optimized_model)
                model_dict[op_name] = optimized_model
                self.__inference_loaders[optimized_model.getInfo("model_name")] = inference_loader

        #self.__models.extend(optimized_models)
        self.__initialized=True
    

    def __checkResidualNormality(self, residuals: Union[np.ndarray, Series]) -> bool:
        """
        Internal function to check Residual Normality using the Shapiro-Wilk test.
        Used to determine if standard ANOVA can be applied.

        Parameters
        ----------
        - residuals: array-like
          The residuals from the OLS model

        Returns
        -------
        - is_normal: bool
          True if residuals are normally distributed (p > 0.05), False otherwise
        """

        logger.debug(f"Here there are our residuals:\n {residuals}")
        shapiro_stat, shapiro_p  = stats.shapiro(residuals)

        logger.debug(f"The results of Shapiro Test-> Shapiro Stats: {shapiro_stat:.4f} | P Value: {shapiro_p:.4f}")

        if shapiro_p < 0.05:
            return False # Reject Null hypothesis (H0: They are from a normal distribution), so not normal
        else:
            return True # Residual are from a Normal distribution

    def __checkResidualHomoschedasticity(self, df: DataFrame, residuals: Union[np.ndarray, Series], normal: bool) -> bool:
        """
        Internal function to check Residual Homoscedasticity (equal variances).
        Uses Bartlett's test if data is normal, otherwise uses Levene's test.

        Parameters
        ----------
        - df: DataFrame
          The results dataframe
        - residuals: array-like
          The residuals from the OLS model
        - normal: bool
          Flag indicating if the residuals are normally distributed

        Returns
        -------
        - is_homoscedastic: bool
          True if variances are equal (p > 0.05), False otherwise
        """

        df['residuals'] = residuals
        groups = [group['residuals'].values for name, group in df.groupby(['Model', 'Optimization'])]

        test_string = "Bartlett" if normal else "Levene"

        if normal:
            stat, p_value = stats.bartlett(*groups)
        else:
            stat, p_value = stats.levene(*groups)

        logger.debug(f"These are the results of {test_string}-> Stat: {stat:.4f} | P Value: {p_value:.4f}")  

        if p_value < 0.05:
            return False
        else:
            return True    

    def __runOneWayAnalysisPerFactor(self, df: DataFrame, factor: str, dv: str='Total model run time', test_type: str='Welch') -> Optional[DataFrame]:
        """
        Internal function to perform a One-Way analysis (Welch ANOVA or Kruskal-Wallis) 
        on a single factor when standard ANOVA assumptions are violated.

        Parameters
        ----------
        - df: DataFrame
          The data to analyze
        - factor: str
          The independent variable (e.g., 'Optimization' or 'Model')
        - dv: str
          The dependent variable (default: 'Total model run time')
        - test_type: str
          'Welch' or 'Kruskal'

        Returns
        -------
        - stats_table: DataFrame
          The resulting statistics table from Pingouin
        """
        
        formula = f'Q("{dv}") ~ C({factor})'
        model = ols(formula, data=df).fit()
        residuals = model.resid

        stats_table = None
        if test_type == 'Welch':
            stats_table = pingu.welch_anova(data=df, dv=dv, between=factor)
        elif test_type == 'Kruskal':
            stats_table = pingu.kruskal(data=df, dv=dv, between=factor)

        return stats_table


    def __runProcessBenchmark(self, aimodel: AIModel, inference_loader: DataLoader) -> Optional[Dict[str, Any]]:
        """
        Executes a benchmark run in a separate subprocess to ensure isolation and accurate timing.

        Parameters
        ----------
        - aimodel: AIModel
          The model to benchmark
        - inference_loader: DataLoader
          The data to use for the benchmark

        Returns
        -------
        - stats: dict
          Dictionary containing performance metrics (Latency, FPS, etc.)
        """

        ctx = get_context("spawn")
        queue = ctx.Queue()

        sub_process_args = (self.__context, aimodel, inference_loader, self.__config_id, queue)
        sub_process = ctx.Process(target=subRunQueue, args=sub_process_args)
        sub_process.start()

        stats = None
        try:
            logger.debug(f"IM TRYING TO DO THE QUEUE GET")
            result_packet = queue.get()
            logger.debug(f"QUEUE GET DONE !!!!")
    
            if result_packet['status'] == "success":
                stats = result_packet['data']
                logger.debug(stats)
            else:
                logger.error(f"Worker reported error: {result_packet.get('message')}")

        except Exception as e:
            logger.error(f"Did not receive data from subprocess (Timeout or Crash): {e}")

        sub_process.join()

        if sub_process.is_alive():
            logger.warning("Subprocess didn't terminate naturally. Forcing termination.")
            sub_process.terminate()

        sub_process.close() # Release file descriptors
        del sub_process
        gc.collect()

        return stats


    def runDesign(self) -> None:
        """
        Executes the experimental design. 
        Iterates through the full factorial list, runs benchmarks for each combination, 
        and saves the raw results to a CSV file.
        """

        assert self.__initialized, "The DoE should be initialized in order to run."

        initialPrint("INFERENCES")

        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Context already set

        temp_dir = PROJECT_ROOT / "temp_results"
        temp_dir.mkdir(exist_ok=True)
    
        results_list = []
        for i, (mod_name, opt_name) in enumerate(self.__design):

            print("\n\t"+ "\x1b[36m" + f" RUN {i+1} / {len(self.__design)}, : {mod_name} | {opt_name}" + "\x1b[37m" + "\n")


            try:
                aimodel = self.__models[mod_name][opt_name]
            except KeyError:
                logger.error(f"Model {mod_name} with Optimization {opt_name} not found!")
                continue

            internal_name = aimodel.getInfo('model_name')
            inference_loader = self.__inference_loaders[internal_name]

            stats = self.__runProcessBenchmark(aimodel, inference_loader)


            row_data = {
                "Model": mod_name,
                "Optimization": opt_name,
            }

            if stats:
                row_data.update(stats)
            else:
                logger.error(f"During single inference something went wrong")
                row_data['Error'] = True

            results_list.append(row_data)
                
        df = DataFrame(results_list)
        doe_result_path = PROJECT_ROOT / "Results" / f"{self.__config_id}" / "DoEResults" / "doe_results_raw.csv"
        doe_result_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(doe_result_path), index = False)

        self.__ran=True

    def runAnova(self) -> None:
        """
        Performs Statistical Analysis on the benchmark results.
        1. Loads raw CSV results.
        2. Fits an OLS model.
        3. Checks assumptions (Normality, Homoscedasticity).
        4. Selects appropriate test (Two-way ANOVA, Welch, or Kruskal-Wallis).
        5. Prints significance results and triggers plot creation.
        """
        #assert self.__ran, "The DoE should be executed with .run before running ANOVA."

        initialPrint("ANOVA ANALYSIS\n")

        file_path = str(PROJECT_ROOT / "Results" / f"{self.__config_id}" / "DoEResults" / "doe_results_raw.csv")
        try:
            df = pd.read_csv(file_path)
            logger.info("Data loaded successfully.")
        except FileNotFoundError:
            logger.critical(f"Error: The file {file_path} was not found.")
            exit(0)
		
        # Columns should be: 'Model', 'Optimization', 'Total_Inference_Time_ms'
        logger.debug(f"Headers of created data [ANOVA TABLE]:")
        logger.debug(df.head())

        logger.info(f"Printing Head of measurements")
        pprint(df.head())

        # Creating Combination Column
        df_plot = deepcopy(df)
        df['Combination'] = df['Model'].astype(str) + "_" + df['Optimization'].astype(str)

        formula = "Q('Total model run time') ~ C(Model) + C(Optimization) + C(Model):C(Optimization)"

        model = ols(formula, data=df).fit()

        ### Check Anova Assumptions ###

        # Normality and Homoschedasticity of Residuals
        is_normal = self.__checkResidualNormality(model.resid)
        is_homo = self.__checkResidualHomoschedasticity(deepcopy(df), model.resid, True)

        anova_table=None
        if is_normal and is_homo:

            logger.info(f"THE DATA ARE NORMAL AND HOMOSCHEDASTIC (EXECUTING F-TEST)")

            # F-test
            anova_table = sm.stats.anova_lm(model, typ=2)
            
        elif is_normal and not is_homo:

            logger.info(f"THE DATA ARE NORMAL BUT NOT HOMOSCHEDASTIC (EXECUTING WELCH TEST)")

            # Welch
            optimization_table = self.__runOneWayAnalysisPerFactor(df, 'Optimization', dv='Total model run time', test_type='Welch')
            model_table = self.__runOneWayAnalysisPerFactor(df, 'Model', dv='Total model run time', test_type='Welch')
            an_table = pingu.welch_anova(data=df, dv='Total model run time', between= 'Combination')

            anova_table = pd.concat([optimization_table, model_table, an_table], ignore_index=True)
        elif not is_normal:

            logger.info(f"THE DATA ARE NOT NORMAL (EXECUTING KRUSKAL-WALLIS TEST)")

            # Kruskal wallis
            optimization_table = self.__runOneWayAnalysisPerFactor(df, 'Optimization', dv='Total model run time', test_type='Kruskal')
            model_table = self.__runOneWayAnalysisPerFactor(df, 'Model', dv='Total model run time', test_type='Kruskal')
            an_table = pingu.kruskal(data=df, dv='Total model run time', between='Combination')

            anova_table = pd.concat([optimization_table, model_table, an_table], ignore_index=True)

        # Visualization: Interaction Plot
        initialPrint("RESULTS\n")
        pprint(anova_table)

        output_path = PROJECT_ROOT / "Results" / f"{self.__config_id}" / "DoEResults" / "anova_results_summary.csv"
        anova_table.to_csv(str(output_path), index=True)
        logger.info(f"Anova Table saved to {output_path}")

        possible_p_cols = ['p-unc', 'PR(>F)', 'p', 'p-corr']
        p_col = next((col for col in possible_p_cols if col in anova_table.columns), None)

        if p_col is None:
            logger.error(f"Error: No p-value column found in this table")
            exit(0)

        print("\n")
        for index, row in anova_table.iterrows():

            factor_name = row['Source'] if 'Source' in row else index
            p_val = row[p_col]

            if math.isnan(p_val):
                continue

            sig_label = f"SIGNIFICANT" if p_val < 0.05 else f"NOT SIGNIFICANT"
            logger.info(f"{str(factor_name):<25} {p_val:.20e}    {sig_label}")

        self.create_plots(df_plot)

    def create_plots(self, df: DataFrame) -> None:
        """
        Function to generate and save interaction plots based on the ANOVA results.

        Parameters
        ----------
        - df: DataFrame
          The result data to plot
        """
        base_path = PROJECT_ROOT / "Results" / f"{self.__config_id}" / "DoEResults"
        plot_path = PROJECT_ROOT / "Results" / f"{self.__config_id}" / "Plots" / "interaction_plot.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        try:

            self.__context.createPlots(df, base_path)

            logger.info(f"PLOTS SAVED TO {base_path}")
        except Exception as e:
            logger.error(f"Generic Error: during plot creation occured this generic error: {e}")



        

if __name__ == "__main__":


    config = {
        "models": [
            {
                "model_name": "resnet18",
                "native": True
            },
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
                "model_name": "efficientnet_b0",
                "native": True
            },  
            {
                'module': 'torchvision.models',
                'model_name': "mnasnet1_0",
                'native': False,
                'weights_path': "ModelData/Weights/mnasnet1_0.pth",
                'device': "cpu",
                'class_name': 'mnasnet1_0',
                'weights_class': 'MNASNet1_0_Weights.DEFAULT',
                'image_size': 224,
                'num_classes': 2,
                "task": "classification",
                'description': 'mnasnet_v2 from torchvision'
            }
        ],
        "optimizations": {
            # "Quantization": {
            #     "method": "QInt8",
            # }, 
            "Pruning": {
                "method": "LnStructured",
                'n': 1, 
                "amount": 0.05,
                "epochs": 1
            },
            "Distillation":{
                "method": True,
                "distilled_paths": {}
            }
        },
        "dataset": {
            "data_dir": "ModelData/Dataset/casting_data",
            "batch_size": 1
        },
        "repetitions": 30,
    }

   # probe = ProbeHardwareManager()

   # there_is_gpu, gpu_type, sys_arch = probe.checkSystem()

    context = PlatformContext()
    config_id = context.createConfigFile(config)
    context.checkDownloadedDependencies()
 

    # pdm = PackageDownloadManager()

    # pdm.checkDownloadedDependencies(there_is_gpu)


    #local scripts
    import BenchmarkingFactory.aiModel as aiModel
    import BenchmarkingFactory.optimization as optimization
    import BenchmarkingFactory.dataWrapper as datawrapper
    from ConfigurationModule.configurationManager import optimizations_library_path, ConfigManager

    #cm = ConfigManager(there_is_gpu=there_is_gpu, arch=sys_arch)

    #config_id = cm.createConfigFile(config)

    doe = DoE(context, config, config_id)

    doe.initializeDoE()
    doe.getContext().initializePlatform(config, config_id)
    doe.runDesign()
    doe.runAnova()

    #doe.getString()




