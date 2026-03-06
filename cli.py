from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import argparse
import json
import sys
import questionary
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from rich.console import Console
from rich.table import Table
from PlatformContext.platform_context import PlatformContext
from BenchmarkingFactory.doe import DoE
from Utils.utilsFunctions import initialPrint

import BenchmarkingFactory.aiModel as aiModel
import BenchmarkingFactory.optimization as optimization
import BenchmarkingFactory.dataWrapper as datawrapper
from ConfigurationModule.configurationManager import optimizations_library_path, ConfigManager

# Import Config Managers
from ConfigurationModule.configurationManager import (
    models_library_path,
    optimizations_library_path
)

PROJECT_ROOT = Path(__file__).resolve().parent

console = Console()


def print_logo():
    """This function prints the logo of MargisBench"""
    
    palette = [
        # (2, 62, 138),  # Warm Blue (#5856d5)
        (0, 119, 182),  # Pan Purple (#7d7be0)
        (0, 150, 199),  # Silver Phoenix (#e9e9f9)
        (0, 180, 216),  # Starlit Night (#3d486f)
        (72, 202, 228),
        (144, 224, 239),
        (173, 232, 244),
        (202, 240, 248),
    ]

    logo = r"""
        __  ___                 _      ____                  __  
       /  |/  /___ _________ _ (_)___ / __ )___  ____  _____/ /_                           
      / /|_/ / __ `/ ___/ __ `/ / ___/ __ / / _ \/ __ \/ ___/ __ \
     / /  / / /_/ / /  / /_/ / (__  ) /_/ //  __/ / / / /__/ / / /    
    /_/  /_/\__,_/_|   \__, /_/____/_____/ \___/_/ /_/\___/_/ /_/         
                      /____/
                      
    """

    lines = logo.splitlines()
    for line in lines:
        for i, char in enumerate(line):
            color_idx = (
                int((i / len(line)) * (len(palette) - 1)) if len(line) > 0 else 0
            )
            r, g, b = palette[color_idx]

            sys.stdout.write(f"\033[38;2;{r};{g};{b}m{char}")
        sys.stdout.write("\033[0m\n")

def get_package_version():
    """Retrieves the version from the installed package metadata."""
    try:
        return version("MargisBench")
    except PackageNotFoundError:
        return "Unknown (package not installed)"

def get_available_opt_per_platform(platform):
    """Gets the available optimization per platform. 
    Given a platform it utilizes the map file (supported_optimizations.json) in the configuration Module"""
    match_plat_otp_path = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles"/ "supported_optimizations.json")
    allowed_keys = None
    match_opti_json = None

    try:
        with open(match_plat_otp_path, "r") as match_opti_file:
            match_opti_json = json.load(match_opti_file)
        allowed_keys = match_opti_json.get(platform, [])
    except (FileNotFoundError, Exception) as e:
        logger.error(f"The path of opti match platform is {match_plat_otp_path}")
        logger.error(f"Error in the retrieving of supported optimizations file: {e}")

    return allowed_keys


def list_options(context):
    """Lists all available models and optimizations in the terminal."""

    platform = context.getPlatform()

    try:
        with open(models_library_path, 'r') as f:
            models = json.load(f)
        with open(optimizations_library_path, 'r') as f:
            optims = json.load(f)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error loading libraries: {e}[/bold red]")
        return

    # Available Optimizations
    allowed_keys = get_available_opt_per_platform(platform)

    filtered_optims = {
        key: optims[key] for key in allowed_keys if key in optims
    }

    # Models Table
    table = Table(title=f"Available Models for {platform}")
    table.add_column("Model Name", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Description")

    for name, details in models.items():
        table.add_row(name, details.get("task", "N/A"), details.get("description", ""))
    
    console.print(table)
    console.print("\n")

    # Optimizations Table
    table_opt = Table(title=f"Available Optimizations for {platform}")
    table_opt.add_column("Type", style="green")
    table_opt.add_column("Methods/Options")

    for opt_type, methods in filtered_optims.items():
        
        if isinstance(methods, list):
            methods_str = ", ".join([str(m) for m in methods])
        else:
            methods_str = str(methods)
        table_opt.add_row(opt_type, methods_str)
    
    console.print(table_opt)



def create_interactive_config(context):
    """Starts an interactive wizard to create a configuration file using ConfigManager."""
    
    platform = context.getPlatform()

    if not platform:
        exit(0)

    # Model choices
    with open(models_library_path, 'r') as f:
        models_lib = json.load(f)
    
    selected_models = questionary.checkbox(
        "Select models to benchmark:",
        choices=list(models_lib.keys())
    ).ask()
    
    if not selected_models:
        console.print("[red]No models selected. Exiting.[/red]")
        exit(0)

    # Optimizations choices
    allowed_opti = get_available_opt_per_platform(platform)

    selected_opts = questionary.checkbox(
        "Select Optimizations to apply:",
        choices = allowed_opti
    ).ask()

    optimizations_dict = {}
    opti_lib_path = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles"/"optimizations_library.json")
    opti_lib_json = None
    try:
        with open(opti_lib_path, "r") as opti_lib_file:
            opti_lib_json = json.load(opti_lib_file)
    except (FileNotFoundError, Exception) as e:
        logger.error(f"The path of opti match platform is {match_plat_otp_path}")
        logger.error(f"Error in the retrieving of supported optimizations file: {e}")



    if "Quantization" in selected_opts:
        initialPrint("QUANTIZATION PARAMETERS")
        q_method = questionary.select(
            "Quantization Bitwidth:",
            choices=opti_lib_json["Quantization"] 
        ).ask()
        

        optimizations_dict["Quantization"] = {
            "method": q_method
        }

    if "Pruning" in selected_opts:
        initialPrint("PRUNING PARAMETERS")
        p_method = questionary.select(
            "Pruning Method:",
            choices=opti_lib_json["Pruning"]
        ).ask()

        p_n = int(questionary.text("Norm degree n (it must be an int):", default="1").ask())
        if p_n <= 0:
            logger.critical(f"Norm degree value not valid")
            exit(0)
        p_amount = float(questionary.text("Pruning Amount (0.0 - 1.0):", default="0.5").ask())
        if p_amount < 0 or p_amount > 1:
            logger.critical(f"P Amount value not valid")
            exit(0)
        p_epochs = int(questionary.text("Fine-tuning Epochs:", default="1").ask())
        if p_epochs <=0:
            logger.critical(f"Number epochs value not valid")
            exit(0)

        optimizations_dict["Pruning"] = {
            "method": p_method,
            "n": p_n,
            "amount": p_amount,
            "epochs": p_epochs
        }

    if "Distillation" in selected_opts:

        optimizations_dict["Distillation"] = {
            "method": True,
            "distilled_paths": {} 
        }


    initialPrint("DATASET PARAMETERS")
    dataset_path = questionary.path("Path to dataset:", default="ModelData/Dataset/casting_data").ask()

    batch_size = int(questionary.text("Batch size:", default="1").ask())
    if batch_size <=0: 
        logger.critical("Batch size value not valid")
        exit(0)

    initialPrint("DOE PARAMETERS")
    repetitions = int(questionary.text("Repetitions:", default="30").ask())
    if repetitions <=1:
        logger.critical("Repetition value not valid, at least 2")
        exit(0)

    config_dict = {
        "models": [models_lib[m] for m in selected_models],
        "optimizations": optimizations_dict,
        "dataset": {
            "data_dir": dataset_path,
            "batch_size": batch_size
        },
        "repetitions": repetitions
    }

    config_id = None
    try:
        console.print("[blue]Validating and generating configuration...[/blue]")
        config_id = context.createConfigFile(config_dict)
        console.print(f"Generated Configuration ID: [bold cyan]{config_id}[/bold cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to create configuration: {e}[/bold red]")

    return config_dict, config_id


def main():
    print_logo()
    # --------- PARSING LOGIC ---------
    parser = argparse.ArgumentParser(description="MargisBench CLI Tool")
    
    parser.add_argument('-c', '--config-path', type=str, nargs='?', const='default', help="Path to the configuration file to execute.\nIf the path is not provided or is executed as 'margis -c default' the fallback is on the file './ConfigurationModule/ConfigFiles/config.json'")
    parser.add_argument('-i', '--interactive', action='store_true', help='Start interactive mode to create a config')
    parser.add_argument('-l', '--list', action='store_true', help='List all available models and optimizations')
    parser.add_argument('-v', '--version', action='store_true', help='Show the version of MargisBench')
    parser.add_argument('-a', '--anova', type=str, nargs='?', const='default', help="Path to the configuration file to execute ANOVA Analysis.\nIf the path is not provided or is executed as 'margis -c default' the fallback is on the file './ConfigurationModule/ConfigFiles/config.json'")


    args = parser.parse_args()

    # -------- COMMAND WITHOUT CONTEXT ---------------


    if (len(sys.argv)==1):
        parser.print_help()
        exit(0)

    if args.version:
        console.print(f"Version: [bold cyan]{get_package_version()}[/bold cyan]")
        return



    # -------- COMMAND WITH CONTEXT ---------------

    initialPrint("PLATFORM")
    context = PlatformContext()


    config_json = None
    config_id = None

    if args.list:
        list_options(context) 
        exit(0)
    elif args.interactive:
        config_json, config_id = create_interactive_config(context) 
    elif args.config_path:
        if args.config_path != "default": 
            path_splitted = args.config_path.split(".")
            if path_splitted[-1] != "json":
                logger.critical("The given path doesn't represent a .json file.")
                exit(1)
            config_json, config_id = context.loadConfigFile(args.config_path)
        else:
            logger.warning("RUNNING EXECUTION WITH DEFAULT CONFIG PATH")
            config_json, config_id = context.loadConfigFile()

    elif args.anova:
        if args.anova != "default": 
            path_splitted = args.anova.split(".")
            if path_splitted[-1] != "json":
                logger.critical("The given path doesn't represent a .json file.")
                exit(1)
            config_json, config_id = context.loadConfigFile(args.anova)
        else:
            logger.warning("RUNNIN ANOVA WITH DEFAULT CONFIG PATH")
            config_json, config_id = context.loadConfigFile()

        if not config_json["platform"] or not config_json["arch"]:
            logger.critical(f"Fournished a wrong configuration. It should be a configuration from a complete execution.")
            exit(1)

        if config_json["platform"] != context.getPlatform():
            logger.critical(f"You should choose the same platform of the provided configuration.")
            exit(1)

        doe = DoE(context, config_json, config_id)
        doe.runAnova()
        exit(0)

    # ------ EXECUTION MODE --------------------

    execution_modes = ['config_path', 'interactive']
    if any(getattr(args, mode) for mode in execution_modes):
        context.checkDownloadedDependencies()
        doe = DoE(context, config_json, config_id)
        doe.initializeDoE()
        doe.getContext().initializePlatform(config_json, config_id)
        doe.runDesign()
        doe.runAnova()



if __name__ == "__main__":
    main()
