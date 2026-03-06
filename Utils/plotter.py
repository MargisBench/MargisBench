from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger

import pandas as pd
import numpy as np
import oapackage
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from abc import ABC, abstractmethod
from statsmodels.graphics.factorplots import interaction_plot

class Plotter(ABC):

    @abstractmethod
    def profile_plot():
        """
        Abstract method for Profiling Plot.

        """
        pass

    @abstractmethod
    def create_plots():
        """
        Abstract Method for plot generation. 

        """
        pass

    def accuracy_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        A barchart that is useful to understand how accuracy changes for each combination of model and optimization.

        Parameters
        ----------
        - df: pd.DataFrame
        The pandas dataframe. 

        - save_path: str
        Destination path of the plot. 


        Returns
        -------
        - None


        """
        df = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()

        df['Accuracy'] = df["Accuracy"]
        
        # Label is just Optimization now, since the Model is in the title

        # Setup Directory
        plot_dir = Path(str(save_path).replace("DoEResults", "Plots")) / "Accuracy"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        unique_optimization = df['Optimization'].unique()
        unique_models = df['Model'].unique()
        num_optimizations = len(unique_optimization)
        
        generated_colors = plt.cm.tab10(np.linspace(0, 1, num_optimizations))

        color_map = dict(zip(unique_optimization, generated_colors))

        sns.set_theme(style="whitegrid")

        # 2. Loop per Model
        unique_models = df['Model'].unique()
        unique_optimization = df['Optimization'].unique()

        

        fig, ax = plt.subplots(figsize=(10, 6)) # Smaller size per plot
        
        indices = range(len(unique_models))
        bar_width = 0.5

        sns.barplot(
            data=df,
            x = 'Model',
            y = 'Accuracy',
            hue= 'Optimization',
            native_scale=True,
            width=0.5,
            legend='auto',
            palette=color_map,
            alpha=0.9
            )

        # 3. Styling
        ax.set_title(f'Accuracy Per Optimization', fontsize=16)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        
        # Set X-ticks
        ax.set_xticks(indices)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_xticklabels(unique_models, rotation=30) # No rotation needed usually for short names
        
        # Legend (only need it once, but good to have on all)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Optimization")
        
        plt.tight_layout()

        # 4. Save Individual Plot
        filename = f"accuracy_per_optimization.png"
        final_path = plot_dir / filename
        plt.savefig(final_path)
        plt.close()

    def heatmap_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        An heatmap that is useful to understand Inference Timeas and Accuracy gains 
        for each combination Model/Optimization.

        Parameters
        ----------
        - df: pd.DataFrame
        The pandas dataframe. 

        - save_path: str
        Destination path of the plot. 


        Returns
        -------
        - None


        """

        pivot_df = df.groupby(['Model', 'Optimization'])['Total model run time'].mean().unstack()
        pivot_accuracy_df = df.groupby(['Model', 'Optimization'])['Accuracy'].mean().unstack()
        

        if 'Base' in pivot_df.columns:
            heatmap_data = pd.DataFrame()
            heatmap_accuracy_data = pd.DataFrame()
            for opt in pivot_df.columns:
                if opt != 'Base':
                    heatmap_data[opt] = ((pivot_df[opt] - pivot_df['Base']) / pivot_df['Base']) * 100
                    heatmap_accuracy_data[opt] = (pivot_accuracy_df[opt]  - pivot_accuracy_df['Base'])
            

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

            #color_map = sns.color_palette("crest", as_cmap=True)
            #color_map_acc = sns.color_palette("crest_r", as_cmap=True)
            
            color_map = "YlOrRd"
            color_map_acc = "YlOrRd_r"

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=color_map, ax=ax1, cbar_kws={'label': 'Inference Time Improvement (%)'})
            sns.heatmap(heatmap_accuracy_data, annot=True, fmt=".2f", cmap=color_map_acc, ax=ax2, cbar_kws={'label': 'Accuracy Improvement (Absolute %)'})
            
            ax1.set_title('Speed Efficiency: (% Inference Time Gain vs Base Model)', fontsize=16)
            ax1.set_ylabel('Model Architecture', fontsize=12)
            ax1.set_xlabel('Optimization Technique', fontsize=12)

            ax2.set_title('Accuracy Cost: (% Absolute Accuracy Gain vs Base Model)', fontsize=16)
            ax2.set_ylabel('')
            ax2.set_xlabel('Optimization Technique', fontsize=12)
            
            plot_path = str(save_path).replace("/DoEResults", "") + "/Plots/optimization_heatmap.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
        
    def pareto_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        An Scatter Plot that is useful to show the "Pareto Curve" composed by the
        best models in terms of FPS and Accuracy.

        Parameters
        ----------
        - df: pd.DataFrame
        The pandas dataframe. 

        - save_path: str
        Destination path of the plot. 


        Returns
        -------
        - None

        """

        x_col, y_col = 'FPS', 'Accuracy'

        df_agg = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()
        unique_models = df_agg['Model'].unique()
        num_models = len(unique_models)
        generated_colors = plt.cm.tab10(np.linspace(0, 1, num_models))

        color_map = dict(zip(unique_models, generated_colors))

        unique_opts = df_agg['Optimization'].unique()
        num_opts = len(unique_opts)
        raw_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

        marker_map = dict(zip(unique_opts, raw_markers[:num_opts]))

        # Fastest first
        sorted_df = df.sort_values(x_col, ascending=False)
        pareto = oapackage.ParetoDoubleLong()
        
        for ii in range(len(df_agg)):
            # Create a vector (point) for the current row
            x_val = df_agg.loc[ii, x_col]
            y_val = df_agg.loc[ii, y_col]

            w = oapackage.doubleVector((x_val, y_val))

            # Check the dominance and add to pareto set
            pareto.addvalue(w, ii)
        
        optimal_indices = list(pareto.allindices())
        pareto_df = df_agg.iloc[optimal_indices].copy()
        pareto_df = pareto_df.sort_values(x_col, ascending=False)
        
        # Plot building
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10,6))

        # 1. Scatter Points
        sns.scatterplot(
            data=df_agg,
            x = x_col,
            y = y_col,
            hue='Model',
            style='Optimization',
            s=150, 
            palette=color_map,
            markers=marker_map,
            alpha=0.9,
            zorder=3 # Points on top
        )

        if not pareto_df.empty:
            # --- RIGHT DELIMITER (Grey) ---
            # Vertical line from the fastest point down to the X-axis
            # Point A: (Max_FPS, 0) -> Point B: (Max_FPS, Acc_at_Max_FPS)
            plt.plot(
                [pareto_df[x_col].iloc[0], pareto_df[x_col].iloc[0]], 
                [0, pareto_df[y_col].iloc[0]],
                color='grey', 
                linestyle='--', 
                linewidth=2,
                zorder=1,
                label='Visual Delimiter'
            )

            # --- LEFT DELIMITER (Grey) ---
            # Horizontal line from the most accurate point left to the Y-axis
            # Point C: (Min_FPS, Max_Acc) -> Point D: (0, Max_Acc)
            plt.plot(
                [pareto_df[x_col].iloc[-1], 0], 
                [pareto_df[y_col].iloc[-1], pareto_df[y_col].iloc[-1]],
                color='grey', 
                linestyle='--', 
                linewidth=2,
                zorder=1
            )

            # --- MAIN PARETO FRONTIER (Yellow) ---
            # Connects the actual model points
            plt.plot(
                pareto_df[x_col], 
                pareto_df[y_col], 
                color='#F1C40F', # Gold/Yellow color
                linestyle='-',   # Solid line to emphasize the "Frontier"
                linewidth=3,
                zorder=2, 
                label='Pareto Frontier'
            )

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Configuration")
        plt.title('Pareto Curve: Accuracy vs Speed', fontsize=16)
        plt.xlabel('Speed (FPS) [Higher is Better]', fontsize=14)
        plt.ylabel('Accuracy (%) [Higher is Better]', fontsize=14)
        plt.tight_layout()

        plot_path = str(save_path).replace("/DoEResults", "")
        plot_path = plot_path + "/Plots" + "/pareto_plot.png"
        plt.savefig(str(plot_path))

    def time_interaction_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        An interaction plot that is useful to understand how the optimization changes the Inference Times. 


        Parameters
        ----------
        - df: pd.DataFrame
        The pandas dataframe. 

        - save_path: str
        Destination path of the plot. 


        Returns
        -------
        - None


        """        

        df_agg = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()
        
        unique_models = df_agg['Model'].unique()
        num_models = len(unique_models)
        
        generated_colors = plt.cm.tab10(np.linspace(0, 1, num_models))
        color_map = dict(zip(unique_models, generated_colors))
        
        raw_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
        marker_map = dict(zip(unique_models, raw_markers[:num_models]))

        # 3. Setup Plotting Theme
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

      
        sns.lineplot(
            data=df_agg,
            x='Optimization',
            y='Total model run time',
            hue='Model',         # Different color per model
            style='Model',       # Different marker per model
            palette=color_map,
            markers=marker_map,
            dashes=False,        # Solid lines only
            markersize=10,       
            linewidth=2
        )

        
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Model")
        
        plt.title('Interaction Plot: Model vs Optimization', fontsize=16)
        plt.xlabel('Optimization Technique', fontsize=14)
        plt.ylabel('Inference Time (ms)', fontsize=14)
        
        plot_path = str(save_path).replace("/DoEResults", "") + "/Plots/interaction_plot.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    def time_box_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        BoxPlots that are useful to understand how the variance of Inference Times 
        for each combination of Model and Optimization.

        Parameters
        ----------
        - df: pd.DataFrame
        The pandas dataframe. 

        - save_path: str
        Destination path of the plot. 


        Returns
        -------
        - None


        """
        save_dir = Path(str(save_path).replace("DoEResults", "Plots")) / "BoxPlots"
        save_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid")

        def add_dense_log_ticks(ax):
            # 1. ALLOW ALL TICKS (1-9) so Matplotlib doesn't panic on small ranges
            # This ensures we get ticks at 6000, 7000, etc. if the data is there.
            locator = ticker.LogLocator(base=10.0, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            ax.yaxis.set_major_locator(locator)
            
            # 2. DEFINITIVE FORMATTER: Force everything to be a plain integer string
            # This ignores Matplotlib's scientific notation logic entirely.
            plain_fmt = ticker.FuncFormatter(lambda x, pos: f'{int(x)}')
            
            ax.yaxis.set_major_formatter(plain_fmt)
            ax.yaxis.set_minor_formatter(plain_fmt) # Apply to minor ticks too, just in case!
            
            # 3. Gridlines
            ax.grid(True, which="minor", color="grey", linewidth=0.3, alpha=0.3)
            ax.grid(True, which="major", color="grey", linewidth=0.8, alpha=0.5)

        # --- PART A: MASTER PLOT (All Models in one) ---
        plt.figure(figsize=(14, 8))
        
        sns.boxplot(
            data=df,
            x='Model',
            y='Total model run time',
            hue='Optimization',
            palette='Set2',
            width=0.7
        )
        
        plt.title('Inference Time Distribution: All Models', fontsize=16)
        plt.ylabel('Inference Time (ms)', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=15)
        plt.legend(title='Optimization', bbox_to_anchor=(1.01, 1), loc='upper left')
        
        plt.yscale('log')
        add_dense_log_ticks(plt.gca()) # Apply the brute-force fixer
        
        plt.tight_layout()
        master_path = save_dir / "all_models_boxplot.png"
        plt.savefig(master_path)
        plt.close() 

        # --- PART B: INDIVIDUAL PLOTS ---
        unique_models = df['Model'].unique()
        
        for model in unique_models:
            model_df = df[df['Model'] == model]
            
            plt.figure(figsize=(8, 6))
            
            sns.boxplot(
                data=model_df,
                x='Optimization',
                hue='Optimization',
                y='Total model run time',
                legend=False,
                width=0.5
            )
            
            sns.stripplot(
                data=model_df,
                x='Optimization',
                y='Total model run time',
                color='black',
                size=4,
                alpha=0.7,
                jitter=True
            )

            plt.title(f'Inference Time Variance: {model}', fontsize=14)
            plt.ylabel('Inference Time (ms)', fontsize=12)
            plt.xlabel('Optimization', fontsize=12)
            
            plt.yscale('log')
            add_dense_log_ticks(plt.gca()) # Apply the brute-force fixer
            
            plt.tight_layout()
            
            filename = f"{model}_boxplot.png"
            file_path = save_dir / filename
            plt.savefig(file_path)
            plt.close()


class PlotterGeneric(Plotter):

    def profile_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        Creates separate stacked bar charts for each model to undestrand the difference
        between Kernel Inference Time and Overhead Inference Time for Generic Platform.

        Parameters
        ----------
        - df: pd.Dataframe
        The pandas Dataframe.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """
        
        df = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()

        df['Stack_Kernel'] = df["Total 'kernel' inference time"]
        df['Stack_Seq_Overhead'] = df["Total sequential executor time"] - df["Total 'kernel' inference time"]
        df['Stack_ORT_Overhead'] = df["Total ONNX runtime overhead"]
        
        df['Label'] = df['Optimization']

        plot_dir = Path(str(save_path).replace("DoEResults", "Plots")) / "Profiling"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_theme(style="whitegrid")

        unique_models = df['Model'].unique()

        for model in unique_models:
            model_df = df[df['Model'] == model].copy()
            
           

            fig, ax = plt.subplots(figsize=(10, 6)) # Smaller size per plot
            
            indices = range(len(model_df))
            bar_width = 0.5

            p1 = ax.bar(
                indices, 
                model_df['Stack_Kernel'], 
                width=bar_width, 
                label='Kernel Time (Compute)', 
                color='#1f77b4', 
                edgecolor='white'
            )

            p2 = ax.bar(
                indices, 
                model_df['Stack_Seq_Overhead'], 
                width=bar_width, 
                bottom=model_df['Stack_Kernel'],
                label='Sequential Executor Overhead', 
                color='#ff7f0e', 
                edgecolor='white'
            )

            bottom_for_p3 = model_df['Stack_Kernel'] + model_df['Stack_Seq_Overhead']
            p3 = ax.bar(
                indices, 
                model_df['Stack_ORT_Overhead'], 
                width=bar_width, 
                bottom=bottom_for_p3, 
                label='ONNX Runtime Overhead', 
                color='#2ca02c', 
                edgecolor='white'
            )

            ax.set_title(f'Profiling Breakdown: {model}', fontsize=16)
            ax.set_ylabel('Time (ms)', fontsize=14)
            ax.set_xlabel('Optimization', fontsize=14)
            
            ax.set_xticks(indices)
            ax.set_xticklabels(model_df['Label'], rotation=0) # No rotation needed usually for short names
            
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Component")
            
            plt.tight_layout()

            # 4. Save Individual Plot
            filename = f"profile_stack_{model}.png"
            final_path = plot_dir / filename
            plt.savefig(final_path)
            plt.close()
            

    def create_plots(self, df, save_path) -> None:
        """
        Creates all the plots for the Generic Platform.
    
        Parameters
        ----------
        - df: pd.Dataframe
        The pandas Dataframe.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """
        self.pareto_plot(df, save_path)
        self.time_interaction_plot(df, save_path)
        self.time_box_plot(df, save_path)
        self.profile_plot(df, save_path)
        self.accuracy_plot(df, save_path)
        self.heatmap_plot(df, save_path)  


class PlotterCoral(Plotter):

    def profile_plot(self, df: pd.DataFrame, save_path: str) -> None:
        """
        Creates separate stacked bar charts for each model to undestrand the difference
        between Kernel Inference Time and Overhead Inference Time for Coral Platform.

        Parameters
        ----------
        - df: pd.Dataframe
        The pandas Dataframe.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """        
        #TODO
        df = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()

        # Stack 1: Initialization Time (The setup cost)
        df['Stack_Init'] = df['Init Time']
        
        # Stack 2: Total Run Time (The inference loop cost)
        df['Stack_Run'] = df['Total model run time']
        
        df['Label'] = df['Optimization']

        # Setup Directory
        plot_dir = Path(str(save_path).replace("DoEResults", "Plots")) / "Profiling"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_theme(style="whitegrid")

        # 2. Loop per Model
        unique_models = df['Model'].unique()

        for model in unique_models:
            # Filter for current model
            model_df = df[df['Model'] == model].copy()

            fig, ax = plt.subplots(figsize=(10, 6))
            
            indices = range(len(model_df))
            bar_width = 0.5

            # --- Stack 1: Total Model Run Time ---
            
            p1 = ax.bar(
                indices, 
                model_df['Stack_Init'], 
                width=bar_width, 
                bottom=model_df['Stack_Run'],
                label='Total Inference Run Time', 
                color='#ff7f0e', 
                edgecolor='white'
            )

            # --- Stack 2: Init Time ---
            # This sits on top of ModelRunTime
            p2 = ax.bar(
                indices, 
                model_df['Stack_Run'], 
                width=bar_width, 
                label='Initialization Time', 
                color='#1f77b4',
                edgecolor='white'
            )


            # 3. Styling
            ax.set_title(f'Profiling Breakdown: {model}', fontsize=16)
            ax.set_ylabel('Time (ms)', fontsize=14)
            ax.set_xlabel('Optimization', fontsize=14)
            
            # Set X-ticks
            ax.set_xticks(indices)
            ax.set_xticklabels(model_df['Label'], rotation=0)
            
            # Legend
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Component")
            
            plt.tight_layout()

            # 4. Save Individual Plot
            filename = f"profile_stack_{model}.png"
            final_path = plot_dir / filename
            plt.savefig(final_path)
            plt.close()        

    def create_plots(self, df: pd.DataFrame, save_path: str)  -> None:
        """
        Creates all the plots for the Coral Platform.
    
        Parameters
        ----------
        - df: pd.DataFrame
        The pandas DataFrame.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """
        self.pareto_plot(df, save_path)
        self.time_interaction_plot(df, save_path)
        self.time_box_plot(df, save_path)
        self.accuracy_plot(df, save_path)
        self.heatmap_plot(df, save_path)  
        self.profile_plot(df, save_path)

class PlotterFusion(Plotter):

    def profile_plot(self, df, save_path):
        """
        Creates separate stacked bar charts for each model to undestrand the difference
        between Kernel Inference Time and Overhead Inference Time for Fusion Platform.

        Parameters
        ----------
        - df: pd.Dataframe
        The pandas Dataframe.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """
        # 1. Preparation
        # Calculate the stack components
        df = df.groupby(['Model', 'Optimization']).mean(numeric_only=True).reset_index()

        df['Stack_Kernel'] = df["Total kernel run time"]
        df['Stack_Seq_Overhead'] = df["Total Overhead"]
        
        # Label is just Optimization now, since the Model is in the title
        df['Label'] = df['Optimization']

        # Setup Directory
        plot_dir = Path(str(save_path).replace("DoEResults", "Plots")) / "Profiling"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_theme(style="whitegrid")

        # 2. Loop per Model
        unique_models = df['Model'].unique()

        for model in unique_models:
            # Filter for current model
            model_df = df[df['Model'] == model].copy()
            
            # Sort by total time (optional, but makes chart cleaner)
            # model_df = model_df.sort_values("Total model run time", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6)) # Smaller size per plot
            
            indices = range(len(model_df))
            bar_width = 0.5

            # --- Stack 1: Kernel ---
            p1 = ax.bar(
                indices, 
                model_df['Stack_Kernel'], 
                width=bar_width, 
                label='Total Kernel Time (Compute)', 
                color='#1f77b4', 
                edgecolor='white'
            )

            # --- Stack 2: Seq Overhead ---
            p2 = ax.bar(
                indices, 
                model_df['Stack_Seq_Overhead'], 
                width=bar_width, 
                bottom=model_df['Stack_Kernel'],
                label='Total Overhead Time', 
                color='#ff7f0e', 
                edgecolor='white'
            )

            # 3. Styling
            ax.set_title(f'Profiling Breakdown: {model}', fontsize=16)
            ax.set_ylabel('Time (ms)', fontsize=14)
            ax.set_xlabel('Optimization', fontsize=14)
            
            # Set X-ticks
            ax.set_xticks(indices)
            ax.set_xticklabels(model_df['Label'], rotation=0) # No rotation needed usually for short names
            
            # Legend (only need it once, but good to have on all)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Component")
            
            plt.tight_layout()

            # 4. Save Individual Plot
            filename = f"profile_stack_{model}.png"
            final_path = plot_dir / filename
            plt.savefig(final_path)
            plt.close()
            

    def create_plots(self, df: pd.DataFrame, save_path: str):
        """
        Creates all the plots for the Fusion Platform.
    
        Parameters
        ----------
        - df: pd.DataFrame
        The pandas DataFrame.
        - save_path: str
        The target path.


        Returns
        -------
        - None

        """
        
        self.pareto_plot(df, save_path)
        self.time_interaction_plot(df, save_path)
        self.time_box_plot(df, save_path)
        self.profile_plot(df, save_path) 
        self.accuracy_plot(df, save_path)
        self.heatmap_plot(df, save_path)  


