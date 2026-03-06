# MargisBench : A Benchmarking Framework
<div align="center">
<img src="../../blob/main/Utils/ProbeHardwareModule/logo.svg" align="center">
<br />
<br />
<img src="https://img.shields.io/badge/python-3.10.19-blue.svg">
<img src="https://img.shields.io/badge/license-MIT-green">
<img src="https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black">
<img src="https://img.shields.io/github/languages/count/MargisBench/MargisBench">
<img src="https://img.shields.io/github/repo-size/MargisBench/MargisBench">
<br />
<br />
</div>



**MargisBench** is a cross-architectural evaluation suite designed to standardize and profile Computer Vision Classification performance across heterogeneous hardware platforms.

It provides a modular framework for running Design of Experiments (DoE) to benchmark AI models on various accelerators, managing everything from model conversion to statistical analysis of inference metrics.

<br />

## 🔎 &nbsp; Main Features 


*   **Cross-Architecture Support**: Run inference on multiple platforms including:
    *   **Generic Platform**: CPU Inference support and possibility to extend it to enable GPU Inference (via ONNX Runtime). 
    *   **Google Coral Dev Board**: Google Edge TPU Inference support. 
    *   **Fusion 844 AI - TTTech**: Hailo8 Accelerator Inference Support.
*   **Optimization Pipeline**: Integrated support for model optimization techniques like **Pruning**, **Quantization** and **Distillation** (it needs distilled pre-trained weights).
*   **Model and Optimization Extensibility**: Support for adding new native/custom Models (available in PyTorch Library) and Optimizations (though implementation from PyTorch Library). 
*   **Design of Experiments (DoE)**: Built-in `BenchmarkingFactory` to orchestrate rigorous statistical experiments, managing repetitions, configurations, and data collection.
*   **Modular Design**: distinctly separated modules for Configuration, Runners, Data Management, and Platform Abstraction.
*   **Simple Setup**: Scripts to initialize project environments and manage platform-specific dependencies.

<br />

## 🛠️ &nbsp; Support 

<br />

> [!NOTE]
>
> The Framework is intended to work on **Linux** and it's tested on **Debian** based distributions (**Raspberry PI OS 13** and **Linux Mint 22.2**) and on **RHEL** distribution (**Fedora 43**). 
The correct behavior on other distributions (other than those mentioned) isn't tested and not guaranteed.

<br />

## 📋 &nbsp; Dependencies & Requirements

### Debian &nbsp;&nbsp; <img src="https://www.debian.org/logos/openlogo-nd.svg" width=25 align="left">
---
```bash
build-essential tk-dev python3-dev python3-pip python3-venv pkg-config gfortran cmake ninja-build cargo
libssl-dev libffi-dev libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libfreetype6-dev liblcms2-dev 
libwebp-dev libopenblas-dev liblapack-dev libsodium-dev protobuf-compiler libprotobuf-dev 
libgl1-mesa-glx libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev
libsqlite3-dev libbz2-dev libexpat1-dev liblzma-dev libdrm-dev git-lfs
```

<br />

### RHEL &nbsp;&nbsp; <img src="https://www.svgrepo.com/show/354273/redhat-icon.svg" width=25 align="left">
---


> It may be needed enable CRB and EPEL repositories to install some of these dependencies. 

```bash
python3-devel python3-pip pkgconfig gcc-gfortran cmake ninja-build cargo 
openssl-devel libffi-devel libsodium-devel ncurses-devel readline-devel 
bzip2-devel sqlite-devel libdb-devel gdbm-devel xz-devel expat-devel zlib-devel 
libjpeg-turbo-devel libpng-devel libtiff-devel freetype-devel lcms2-devel libwebp-devel 
openblas-devel lapack-devel protobuf-devel protobuf-compiler 
mesa-libGL-devel libdrm-devel tk-devel git-lfs
```

<br />

>[!NOTE]
>
> To execute the framework on `Fusion 844 AI` target, the latest version of Docker is required to perform the [Model Conversion](../../wiki/Converters#2-fusion-converter) pipeline.


### System Requirements
- x86_64 or aarch64 architecture and 64-bit Operating System,
- At least 8Gb of **RAM**,
- At least 15Gb of **Disk**.

<br />

**Python Installation**

The framework is fully implemented and tested on [**Python 3.10.19**](https://www.python.org/downloads/release/python-31019/). It's reccommended to install it via the system package manager
or build it from source.

<br />

## 🛠️ &nbsp; MargisBench Installation

MargisBench includes an initialization script to set up the project files, setup the virtual environment and install necessary dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MargisBench/MargisBench.git
    cd MargisBench
    git lfs pull
    ```

2.  **Run the initialization script:**
    This sets up the `./PackageDownloadModule/requirementsFileDirectory/.installed.json` status, cleans old environments, and creates a fresh `venv` with all requirements.
    ```bash
    chmod +x initialize_project.sh
    ./initialize_project.sh
    ```

3.  **Activate the Virtual Environment:**
    ```bash
    source venv/bin/activate
    ```
<br />

> [!WARNING]
>
> On RHEL distributions provided with SELinux, some error can occur due to *executable stack flag bit* set on some dependency package. To remove this flag temporarily the following command can be executed: `execstack -c <path-to-package-in-venv>`. To know more about this visit [Execstack Flag in RHEL SELinux-enforced distributions](../../wiki/Possible-Problems#execstack-flag-in-rhel-selinux-enforced-distributions) section in the wiki.  

<br />

## 📂 &nbsp; Project Structure

The framework is organized into specific packages to handle different aspects of the benchmarking pipeline:

*   **`BenchmarkingFactory/`**: The core of the framework. Contains the `DoE` (Design of Experiments) manager, the `AIModel` definition, `DataWrapper` for dataset handling, and `Optimization` logic.
*   **`ConfigurationModule/`**: Handles parsing and validating configuration files (JSON schemas) for devices and experiments. It also contains the `ConfigFiles/` folder which contains all the configuration files.  
*   **`Converters/`**: Contains logic to convert standard models into hardware-specific formats (e.g., for Coral to .tflie or Fusion/Hailo devices to .hef).
*   **`ModelData/`**: Directory structure for storing Model Weights, Datasets, and generated ONNX models.
*   **`PackageDownloadModule/`**: Manages the download and installation of platform-specific dependencies.
*   **`PlatformContext/`**: Abstraction layer for the underlying platform context.
*   **`PlatformInitializers/`**: Scripts and utilities to prepare and initialize specific hardware platforms (e.g., *pushing files to edge devices*, *enabling performance mode etc.*).
*   **`Runner/`**: Contains the `RunnerModule` implementations that execute the actual inferences on the target hardware.
*   **`Utils/`**: General utility functions, logging configurations, and statistical helper tools.
*   **`Results/`**: Folder were the final results will be saved.

More about this in the dedicated [Wiki](../../wiki). **Check it out!**

<br />

## ⚙️ &nbsp; Usage

> [!NOTE]
> The framework is based on [Casting Products Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) that is already fournished inside the project. 
>
>The native models are pre-trained on this dataset. 
>
>Other dataset can be used as long as they are compliant to this [folder structure](https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks:~:text=PNG%2E-,Folder%20Structure%20Example) and the relative `.pth` for the specific models are provided in `ModelData/Weights` folder.


<br />

The framework typically operates by defining a configuration dictionary (or JSON file) that specifies the models, optimizations, and datasets to be tested. This configuration can be **loaded** from **`ConfigurationModule/ConfigFiles/config.json`** or created in an **interactive way**.

The `DoE` class in `BenchmarkingFactory/doe.py` acts as the orchestrator. 

<br />


### **📝 Configuration Structure Example**


> [!NOTE]
>
>To know more about configurations composition, feature and limits please check [About Configuration Files](../../wiki/About-Configuration-Files) wiki page. If you are willing to compile configurations file manually, read the mentioned section carefully. 


<br />

This is a configuration example that can be provided to the tool. This configuration can be **loaded** from a json file or **created** in an interactive way. 

```json
{
    "models": [
        {
            "model_name": "resnet18",
            "native": true
        },
        {
            "module": "torchvision.models",
            "model_name": "mobilenet_v2",
            "native": false,
            "weights_path": "ModelData/Weights/mobilenet_v2.pth",
            "device": "cpu",
            "qat_mode": false,
            "class_name": "mobilenet_v2",
            "weights_class": "MobileNet_V2_Weights.DEFAULT",
            "image_size": 224,
            "num_classes": 2,
            "task": "classification",
            "description": "Mobilenet V2 from torchvision"
        }
    ],
    "optimizations": {
        "Quantization": {
              "method": "QInt8"
        },
        "Pruning": {
            "method": "LnStructured",
            "n": 1,
            "amount": 0.05,
            "epochs": 1
        },
        "Distillation": {
            "method": true,
            "distilled_paths": {}
        }
    },
    "dataset": {
        "data_dir": "ModelData/Dataset/casting_data",
        "batch_size": 1
    },
    "repetitions": 30
}
```

Thanks to this configuration, the tool will perform inferences with **resnet18** (Quantized, Optimized, Distilled), and **mobilenet_v2** (Quantized, Optimized, Distilled) on dataset `ModelData/Dataset/casting_data` with *batch size* 1. 
Every model configuration (Model + Optimization) will be executed `repetitions` times (30 in this case). 

<br />

### **👨‍💻 MargisBench Command Line Interface**
After executed all the steps in [MargisBench Installation](#%EF%B8%8F--margisbench-installation) section, the framework provides a command line interface callable with `margis` command.

Show **helper**:
```bash
margis
```
or 
```bash
margis -h 
```

<img src="https://i.imgur.com/XSWfL0f.png">

List **available *native* models and optimization for specific platform**:

```bash
margis -l
```

<img src="https://i.imgur.com/043hxd0.png">

This functionality is helpful to understand the target platform which types of optimization supports. (e. g. for **Coral Dev Board** and **Fusion 844 AI** the **Quantization** optimization is not allowed in the configuration because all the models will be directly quantized in order to run the Inference on the accelerators). 

**How perform a Benchmark**

>[!NOTE]
>
> In every execution scenario, the user will have to choose the target platform. 

<br />

#### **Configuration File Mode**

```bash
margis -c 
```
or
```bash
margis -c default
```

In the default way, **MargisBench** will load the configuration from `ConfigurationModule/ConfigFiles/config.json`. 
The `ConfigurationModule/ConfigFiles/config.json` file will be overwritten at every execution of a *new* configuration (e. g. with a different file provided by **config-path flag** or created with **interactive** mode), therefore it represents the last configuration provided to the framework that was correctly parsed. 

The configuration file provided should be similar to that shown in [Configuration Structure Example](#-configuration-structure-example) section.

>[!WARNING]
>
> If the `ConfigurationModule/ConfigFiles/config.json`, an *error* will occur. In this case it's recommended to give a `.json` file in input, compliant with the configuration schema, or create a new one through the interactive mode. 

To provide an external configuration file
```bash
margis -c <path-to-json-config-file>
```
The configuration should be compliant to the [JSON Schema](../../wiki/About-Configuration-Files) which is specific to the chosen platform. 


#### **Interactive Mode**
>[!NOTE]
>
>The interactive mode provides a simple way to generate a configuration with **native** models that are already present in the project (trained on [Casting Products Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)). In order to use a configuration with **custom** models, the file must be compiled manually and provided through [Configuration File Mode](#configuration-file-mode). 

```bash
margis -i 
```
This command will show a custom initialization wizard, that helps the user to choose *models*, *available optimizations* and their parameters  (for that specific chosen platform) and the *dataset path*. 

<img src="https://i.imgur.com/WFcgboW.png">

 
### **📊 MargisBench Results**

After the execution, the framework will perform an **ANOVA** analysis on the collected data during the inferences and some visual plot are generated under `Results/config_id/` where *config_id* is the configuration ID generated from the provided configuration (plus two arguments added during the execution: `platform` and `arch`) . 

<img src="https://i.imgur.com/ny7eyKe.png">

The structure of `Results/config_id` should be like the following:

```bash
Results/f71d264fc991_generic/
├── DoEResults
│   ├── anova_results_summary.csv
│   └── doe_results_raw.csv
└── Plots
    ├── Accuracy
    │   └── accuracy_per_optimization.png
    ├── BoxPlots
    │   ├── all_models_boxplot.png
    │   ├── efficientnet_b0_boxplot.png
    │   ├── mnasnet1_0_boxplot.png
    │   ├── mobilenet_v2_boxplot.png
    │   └── resnet18_boxplot.png
    ├── interaction_plot.png
    ├── optimization_heatmap.png
    ├── pareto_plot.png
    └── Profiling
        ├── profile_stack_efficientnet_b0.png
        ├── profile_stack_mnasnet1_0.png
        ├── profile_stack_mobilenet_v2.png
        └── profile_stack_resnet18.png
```

Consult the [Plot Examples Section](../../wiki/Plotter#plot-examples) for some plot examples. 

**How perform ANOVA Analysis with MargisBenchCLI**

In order to execute *only* the **ANOVA** analysis and show the results on terminal (from existing data) the flag `-a` can be used. 

```bash
margis -a
```
or 
```bash
margis -a default
```
or 
```bash
margis -a <path-to-json-config-file>
```

The configuration file will be loaded (`ConfigurationModule/ConfigFiles/config.json` in the default case) and the `config_id` is calculated. If the file `Results/config_id/DoEResultsdoe_results_raw.csv` exists the analysis will be performed and visual plots/anova_results_summary.csv recreated.



<br />

## 🫂 Credits

This section mentions the main open source projects used to gave life to **MargisBench**. Thank you guys!

- [ONNX](https://onnx.ai/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [onnx2tf](https://github.com/PINTO0309/onnx2tf)
- [pycoral](https://github.com/google-coral/pycoral)
- [hailort](https://github.com/hailo-ai/hailort)
- [Casting Products Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)


<br />

## 📜 License 

This project is licensed under the MIT License - see the [LICENSE](../../blob/main/LICENSE) file for details.
