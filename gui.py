import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import time
import os
from ga import GeneticAlgorithm
from functions_plots import ackley, rastrigin
from analysis import compute_summary, plot_boxplot_for_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ConfigFrame:
    def __init__(self, parent, run_callback, run_30_callback, update_crossover_callback):
        """
        Initialize configuration frame for GA parameters and run buttons.
        Parameters:
            parent: Parent Tkinter frame
            run_callback: Callback for running GA once
            run_30_callback: Callback for running GA 30 times
            update_crossover_callback: Callback to update crossover options
        """
        self.frame = ttk.Frame(parent)
        self.run_callback = run_callback
        self.run_30_callback = run_30_callback
        self.update_crossover_callback = update_crossover_callback
        self._setup_ga_parameters()
        self._setup_buttons()

    def _setup_ga_parameters(self):
        """Set up GA parameter input fields"""
        # Function selection
        tk.Label(self.frame, text="Function:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.function_var = tk.StringVar(value="Ackley")
        self.function_combo = ttk.Combobox(self.frame, textvariable=self.function_var, values=["Ackley", "Rastrigin"])
        self.function_combo.grid(row=0, column=1, padx=5, pady=5)

        # Encoding option
        tk.Label(self.frame, text="Encoding:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.encoding_var = tk.StringVar(value="real")
        self.encoding_combo = ttk.Combobox(self.frame, textvariable=self.encoding_var, values=["binary", "real"])
        self.encoding_combo.grid(row=1, column=1, padx=5, pady=5)
        self.encoding_combo.bind('<<ComboboxSelected>>', self.update_crossover_callback)

        # Crossover option
        tk.Label(self.frame, text="Crossover:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.crossover_var = tk.StringVar(value="arithmetic")
        self.crossover_combo = ttk.Combobox(self.frame, textvariable=self.crossover_var, values=["arithmetic", "blx-alpha"])
        self.crossover_combo.grid(row=2, column=1, padx=5, pady=5)

        # Population size
        tk.Label(self.frame, text="Population Size:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.pop_size_var = tk.StringVar(value="500")
        tk.Entry(self.frame, textvariable=self.pop_size_var).grid(row=3, column=1, padx=5, pady=5)

        # Generations
        tk.Label(self.frame, text="Generations:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        self.generations_var = tk.StringVar(value="100")
        tk.Entry(self.frame, textvariable=self.generations_var).grid(row=4, column=1, padx=5, pady=5)

        # Mutation rate
        tk.Label(self.frame, text="Mutation Rate (0-1):").grid(row=5, column=0, padx=5, pady=5, sticky='e')
        self.mutation_var = tk.StringVar(value="0.05")
        tk.Entry(self.frame, textvariable=self.mutation_var).grid(row=5, column=1, padx=5, pady=5)

        # Crossover rate
        tk.Label(self.frame, text="Crossover Rate (0-1):").grid(row=6, column=0, padx=5, pady=5, sticky='e')
        self.crossover_rate_var = tk.StringVar(value="0.8")
        tk.Entry(self.frame, textvariable=self.crossover_rate_var).grid(row=6, column=1, padx=5, pady=5)

    def _setup_buttons(self):
        """Set up the run buttons"""
        tk.Button(self.frame, text="Run GA Once", command=self.run_callback).grid(row=7, column=0, padx=5, pady=10)
        tk.Button(self.frame, text="Run GA 30 Times", command=self.run_30_callback).grid(row=7, column=1, padx=5, pady=10)

    def get_parameters(self):
        """Return GA parameters as a dictionary"""
        return {
            'function_name': self.function_var.get(),
            'encoding': self.encoding_var.get(),
            'crossover': self.crossover_var.get(),
            'pop_size': self.pop_size_var.get(),
            'num_generations': self.generations_var.get(),
            'mutation_rate': self.mutation_var.get(),
            'crossover_rate': self.crossover_rate_var.get()
        }

class StatsFrame:
    def __init__(self, parent, function_var):
        """
        Initialize statistics frame with scrollable text area.
        Parameters:
            parent: Parent Tkinter frame
            function_var: StringVar for selected function
        """
        self.frame = ttk.Frame(parent)
        self.function_var = function_var
        tk.Label(self.frame, text="Summary Statistics").grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        self.stats_inner_frame = ttk.Frame(self.frame)
        self.stats_inner_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        self.summary_text = tk.Text(self.stats_inner_frame, height=10, width=50, wrap='none')
        self.summary_text.grid(row=0, column=0, sticky='nsew')
        self.scrollbar = ttk.Scrollbar(self.stats_inner_frame, orient='vertical', command=self.summary_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.summary_text['yscrollcommand'] = self.scrollbar.set

    def update_summary(self):
        """Update the scrollable summary statistics table for the selected function"""
        self.summary_text.delete(1.0, tk.END)
        if os.path.exists('ga_results.csv'):
            df = pd.read_csv('ga_results.csv')
            function_name = self.function_var.get()
            df_filtered = df[df['Function'] == function_name]
            if not df_filtered.empty:
                summary = compute_summary(df_filtered)
                header = f"{'Encoding':<8} {'Crossover':<12} {'Mean':<8} {'Std':<8}\n"
                self.summary_text.insert(tk.END, header)
                self.summary_text.insert(tk.END, "-" * 36 + "\n")
                for _, row in summary.iterrows():
                    line = f"{row['Encoding']:<8} {row['Crossover'][:11]:<12} {row['mean']:<8.4f} {row['std']:<8.4f}\n"
                    self.summary_text.insert(tk.END, line)
            else:
                self.summary_text.insert(tk.END, f"No results for {function_name}.\nRun the GA to generate data.")
        else:
            self.summary_text.insert(tk.END, "No results available.\nRun the GA to generate data.")

class VisualizationFrame:
    def __init__(self, parent, function_var):
        """
        Initialize visualization frame with box plot and function plots.
        Parameters:
            parent: Parent Tkinter frame
            function_var: StringVar for selected function
        """
        self.frame = ttk.Frame(parent)
        self.function_var = function_var
        self._setup_boxplot_frame()
        self._setup_function_plots_frame()

    def _setup_boxplot_frame(self):
        """Set up the box plot frame"""
        self.boxplot_frame = ttk.Frame(self.frame)
        self.boxplot_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.fig_boxplot, self.ax_boxplot = plt.subplots(figsize=(6, 4))
        self.canvas_boxplot = FigureCanvasTkAgg(self.fig_boxplot, master=self.boxplot_frame)
        self.canvas_boxplot.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

    def _setup_function_plots_frame(self):
        """Set up the function plots frame with contour and surface plots"""
        self.plot_frame = ttk.Frame(self.frame)
        self.plot_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.fig_contour, self.ax_contour = plt.subplots(figsize=(4, 4))
        self.canvas_contour = FigureCanvasTkAgg(self.fig_contour, master=self.plot_frame)
        self.canvas_contour.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.fig_surface, self.ax_surface = plt.subplots(figsize=(4, 4), subplot_kw={'projection': '3d'})
        self.canvas_surface = FigureCanvasTkAgg(self.fig_surface, master=self.plot_frame)
        self.canvas_surface.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

    def update_boxplot(self):
        """Update the box plot using plot_boxplot from analysis.py"""
        self.ax_boxplot.clear()
        if os.path.exists('ga_results.csv'):
            df = pd.read_csv('ga_results.csv')
            function_name = self.function_var.get()
            plot_boxplot_for_gui(df, function_name, self.ax_boxplot)
            self.ax_boxplot = plt.gca()  # Update ax_boxplot to current axes
            self.fig_boxplot.tight_layout()
        else:
            self.ax_boxplot.text(0.5, 0.5, 'No data', ha='center', va='center')
        self.canvas_boxplot.draw()

    def update_function_plots(self):
        """Update side-by-side 2D contour and 3D surface plots"""
        function_name = self.function_var.get()
        func = ackley if function_name == "Ackley" else rastrigin
        x_range = (-10, 10) if function_name == "Ackley" else (-5, 5)
        y_range = x_range
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        self.ax_contour.clear()
        contour = self.ax_contour.contourf(X, Y, Z, levels=50, cmap='viridis')
        if not hasattr(self, 'colorbar_contour'):
            self.colorbar_contour = self.fig_contour.colorbar(contour, ax=self.ax_contour)
        else:
            self.colorbar_contour.update_normal(contour)
        self.ax_contour.set_title(f'{function_name} Contour')
        self.ax_contour.set_xlabel('x')
        self.ax_contour.set_ylabel('y')
        self.canvas_contour.draw()

        self.ax_surface.clear()
        surf = self.ax_surface.plot_surface(X, Y, Z, cmap='viridis')
        if not hasattr(self, 'colorbar_surface'):
            self.colorbar_surface = self.fig_surface.colorbar(surf, ax=self.ax_surface, shrink=0.5, aspect=5)
        else:
            self.colorbar_surface.update_normal(surf)
        self.ax_surface.set_title(f'{function_name} Surface')
        self.ax_surface.set_xlabel('x')
        self.ax_surface.set_ylabel('y')
        self.ax_surface.set_zlabel('f(x, y)')
        self.canvas_surface.draw()

class GARunner:
    def __init__(self, result_label):
        """
        Initialize GA runner with result label for updates.
        Parameters:
            result_label: Tkinter Label to display results
        """
        self.result_label = result_label
        self.results = []
        if os.path.exists('ga_results.csv'):
            self.results = pd.read_csv('ga_results.csv').to_dict('records')

    def run_ga(self, params, seed=None):
        """Run the genetic algorithm with given parameters"""
        try:
            function_name = params['function_name']
            encoding = params['encoding']
            crossover = params['crossover']
            pop_size = int(params['pop_size'])
            num_generations = int(params['num_generations'])
            mutation_rate = float(params['mutation_rate'])
            crossover_rate = float(params['crossover_rate'])

            if not (0 <= mutation_rate <= 1 and 0 <= crossover_rate <= 1):
                raise ValueError("Rates must be between 0 and 1")
            if pop_size < 10 or num_generations < 10:
                raise ValueError("Population size and generations must be >= 10")

            function = ackley if function_name == "Ackley" else rastrigin
            domain = (-10, 10) if function_name == "Ackley" else (-5, 5)

            seed = seed if seed is not None else int(time.time() * 1000) % 2**32

            ga = GeneticAlgorithm(function, domain, encoding, crossover, pop_size, num_generations, mutation_rate, crossover_rate, seed)
            best_solution, best_fitness = ga.run()

            self.result_label.config(text=f"Best Solution: {best_solution}\nBest Fitness: {best_fitness:.6f}")

            self.results.append({
                'Function': function_name,
                'Encoding': encoding,
                'Crossover': crossover,
                'Population': pop_size,
                'Generations': num_generations,
                'Mutation_Rate': mutation_rate,
                'Crossover_Rate': crossover_rate,
                'Best_Fitness': best_fitness,
                'Best_Solution': str(best_solution),
                'Seed': seed
            })
            pd.DataFrame(self.results).to_csv('ga_results.csv', index=False)

            return best_solution, best_fitness

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def run_ga_30_times(self, params):
        """Run the genetic algorithm 30 times"""
        try:
            self.result_label.config(text="Running 30 times... Please wait.")
            for i in range(30):
                seed = int(time.time() * 1000 + i) % 2**32
                self.run_ga(params, seed=seed)
            self.result_label.config(text="Completed 30 runs!\nResults saved to ga_results.csv")
            messagebox.showinfo("Success", "30 runs completed. Results saved to ga_results.csv")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during 30 runs: {str(e)}")

class GAInterface:
    def __init__(self, root):
        """
        Initialize Tkinter GUI to coordinate configuration, statistics, and visualization.
        Parameters:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Genetic Algorithm Optimizer")
        self._setup_scrollable_layout()
        self.result_label = tk.Label(self.config_frame, text="Run the GA to see results")
        self.result_label.grid(row=8, column=0, columnspan=2, padx=5, pady=5)
        self.ga_runner = GARunner(self.result_label)
        self.config = ConfigFrame(self.config_frame, self.run_ga, self.run_ga_30_times, self.update_crossover_options)
        self.config.frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='n')
        self.stats = StatsFrame(self.config_frame, self.config.function_var)
        self.stats.frame.grid(row=10 , column=0, columnspan=2, padx=5, pady=5, sticky='n')

        self.stats.update_summary()
        self.visualization = VisualizationFrame(self.display_frame, self.config.function_var)
        self.visualization.frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
        self.visualization.update_function_plots()
        self.visualization.update_boxplot()

    def _setup_scrollable_layout(self):
        """Set up the scrollable main layout with canvas and scrollbars"""
        self.canvas = tk.Canvas(self.root)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.v_scrollbar = ttk.Scrollbar(self.root, orient='vertical', command=self.canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.h_scrollbar = ttk.Scrollbar(self.root, orient='horizontal', command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame = ttk.Frame(self.canvas)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.main_frame, anchor='nw')
        self.main_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.config_frame = ttk.Frame(self.main_frame)
        self.config_frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.grid(row=0, column=1, padx=5, pady=5, sticky='n')

    def update_crossover_options(self, event=None):
        """Update the crossover options based on the selected encoding"""
        encoding = self.config.encoding_var.get()
        if encoding == 'binary':
            self.config.crossover_combo['values'] = ["1-point", "2-point"]
            self.config.crossover_var.set("1-point")
        else:
            self.config.crossover_combo['values'] = ["arithmetic", "blx-alpha"]
            self.config.crossover_var.set("arithmetic")
        self.update_plots()

    def update_plots(self, event=None):
        """Update all visualizations"""
        self.visualization.update_function_plots()
        self.stats.update_summary()
        self.visualization.update_boxplot()

    def run_ga(self):
        """Run the GA once with current parameters"""
        params = self.config.get_parameters()
        self.ga_runner.run_ga(params)
        self.update_plots()

    def run_ga_30_times(self):
        """Run the GA 30 times with current parameters"""
        params = self.config.get_parameters()
        self.ga_runner.run_ga_30_times(params)
        self.update_plots()

if __name__ == "__main__":
    root = tk.Tk()
    app = GAInterface(root)
    root.mainloop()