import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

def load_results(file_path='ga_results.csv'):
    """
    Load GA results from CSV.
    Returns: DataFrame
    """
    return pd.read_csv(file_path)

def compute_summary(df):
    """
    Compute mean and std of the best fitness for each configuration.
    Parameters:
        df: DataFrame with GA results
    Returns: DataFrame with mean and std
    """
    return df.groupby(['Function', 'Encoding', 'Crossover'])['Best_Fitness'].agg(['mean', 'std']).reset_index()

def generate_boxplot_data(df, function_name=None):
    """
    Generate data for box plot of fitness distributions, optionally for a specific function.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley', 'Rastrigin', or None for all functions
    Returns: Tuple of (data, labels, colors, positions)
    """
    functions = [function_name] if function_name else ['Ackley', 'Rastrigin']
    encodings = ['binary', 'real']
    crossovers = {'binary': ['1-point', '2-point'], 'real': ['arithmetic', 'blx-alpha']}
    colors = ['#1f77b4', '#ff7f0e']  # Blue for binary, orange for real

    data = []
    labels = []
    color_list = []
    positions = []
    pos = 1

    for func in functions:
        for crossover in crossovers['binary'] + crossovers['real']:
            for enc in encodings:
                if (enc == 'binary' and crossover in ['1-point', '2-point']) or \
                        (enc == 'real' and crossover in ['arithmetic', 'blx-alpha']):
                    fitness = df[(df['Function'] == func) & (df['Encoding'] == enc) & (df['Crossover'] == crossover)]['Best_Fitness']
                    if not fitness.empty:
                        data.append(fitness.values)
                        labels.append(f'{crossover}\n{enc}')
                        color_list.append(colors[0] if enc == 'binary' else colors[1])
                        positions.append(pos)
                        pos += 1
            pos += 0.5  # Space between crossover groups

    return data, labels, color_list, positions

def plot_boxplot_to_file(df, function_name, save_path):
    """
    Generate box plot for a specific function's fitness distributions and save to file.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley' or 'Rastrigin'
        save_path: Path to save the plot
    """
    df_func = df[df['Function'] == function_name]
    data, labels, colors, positions = generate_boxplot_data(df_func, function_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Fitness Distribution for {function_name}')
    ax.set_xlabel('Crossover and Encoding')
    ax.set_ylabel('Best Fitness')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='binary'),
                       Patch(facecolor=colors[1], label='real')]
    ax.legend(handles=legend_elements, title='Encoding')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxplot_for_gui(df, function_name, ax):
    """
    Generate box plot for a specific function's fitness distributions for GUI display.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley' or 'Rastrigin'
        ax: Matplotlib axes object for GUI
    """
    df_func = df[df['Function'] == function_name]
    data, labels, colors, positions = generate_boxplot_data(df_func, function_name)
    ax.clear()
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Fitness Distribution for {function_name}')
    ax.set_xlabel('Crossover and Encoding')
    ax.set_ylabel('Best Fitness')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='binary'),
                       Patch(facecolor=colors[1], label='real')]
    ax.legend(handles=legend_elements, title='Encoding')

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

def load_results(file_path='ga_results.csv'):
    """
    Load GA results from CSV.
    Returns: DataFrame
    """
    return pd.read_csv(file_path)

def compute_summary(df):
    """
    Compute mean and std of the best fitness for each configuration.
    Parameters:
        df: DataFrame with GA results
    Returns: DataFrame with mean and std
    """
    return df.groupby(['Function', 'Encoding', 'Crossover'])['Best_Fitness'].agg(['mean', 'std']).reset_index()

def generate_boxplot_data(df, function_name=None):
    """
    Generate data for box plot of fitness distributions, optionally for a specific function.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley', 'Rastrigin', or None for all functions
    Returns: Tuple of (data, labels, colors, positions)
    """
    functions = [function_name] if function_name else ['Ackley', 'Rastrigin']
    encodings = ['binary', 'real']
    crossovers = {'binary': ['1-point', '2-point'], 'real': ['arithmetic', 'blx-alpha']}
    colors = ['#1f77b4', '#ff7f0e']  # Blue for binary, orange for real

    data = []
    labels = []
    color_list = []
    positions = []
    pos = 1

    for func in functions:
        for crossover in crossovers['binary'] + crossovers['real']:
            for enc in encodings:
                if (enc == 'binary' and crossover in ['1-point', '2-point']) or \
                        (enc == 'real' and crossover in ['arithmetic', 'blx-alpha']):
                    fitness = df[(df['Function'] == func) & (df['Encoding'] == enc) & (df['Crossover'] == crossover)]['Best_Fitness']
                    if not fitness.empty:
                        data.append(fitness.values)
                        labels.append(f'{crossover}\n{enc}')
                        color_list.append(colors[0] if enc == 'binary' else colors[1])
                        positions.append(pos)
                        pos += 1
            pos += 0.5  # Space between crossover groups

    return data, labels, color_list, positions

def plot_boxplot_to_file(df, function_name, save_path):
    """
    Generate box plot for a specific function's fitness distributions and save to file.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley' or 'Rastrigin'
        save_path: Path to save the plot
    """
    df_func = df[df['Function'] == function_name]
    data, labels, colors, positions = generate_boxplot_data(df_func, function_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Fitness Distribution for {function_name}')
    ax.set_xlabel('Crossover and Encoding')
    ax.set_ylabel('Best Fitness')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='binary'),
                       Patch(facecolor=colors[1], label='real')]
    ax.legend(handles=legend_elements, title='Encoding')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxplot_for_gui(df, function_name, ax):
    """
    Generate box plot for a specific function's fitness distributions for GUI display.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley' or 'Rastrigin'
        ax: Matplotlib axes object for GUI
    """
    df_func = df[df['Function'] == function_name]
    data, labels, colors, positions = generate_boxplot_data(df_func, function_name)
    ax.clear()
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Fitness Distribution for {function_name}')
    ax.set_xlabel('Crossover and Encoding')
    ax.set_ylabel('Best Fitness')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='binary'),
                       Patch(facecolor=colors[1], label='real')]
    ax.legend(handles=legend_elements, title='Encoding')

def perform_ttest(fitness1, fitness2, name1, name2, result_key):
    """
    Perform t-test between two fitness arrays and return formatted results.
    Parameters:
        fitness1: First fitness array (e.g., 1-point or arithmetic)
        fitness2: Second fitness array (e.g., 2-point or blx-alpha)
        name1: Name of first group (e.g., '1-point', 'arithmetic')
        name2: Name of second group (e.g., '2-point', 'blx-alpha')
        result_key: Key for result dictionary (e.g., '1-point_vs_2-point')
    Returns: Dict with t-test results
    """
    result = {
        't_stat': None,
        'p_value': None,
        'significant': False,
        f'sample_size_{name1}': len(fitness1),
        f'sample_size_{name2}': len(fitness2),
        f'mean_{name1}': np.mean(fitness1) if len(fitness1) > 0 else None,
        f'mean_{name2}': np.mean(fitness2) if len(fitness2) > 0 else None,
        'error': None
    }

    if len(fitness1) > 1 and len(fitness2) > 1:
        if np.var(fitness1) == np.var(fitness2):
            mean_diff = np.mean(fitness1) - np.mean(fitness2)
            result.update({
                'p_value': 0.0 if mean_diff != 0 else 1.0,
                'significant': mean_diff != 0,
                'error': f'Equal variance in both {name1} and {name2}; compared means directly'
            })
        else:
            t_stat, p_value = ttest_ind(fitness1, fitness2, equal_var=False)
            result.update({
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'error': None
            })
    else:
        result['error'] = f'Insufficient data for {result_key} comparison (< 2 samples)'

    return result

def ttest_compare_encodings(df, function_name):
    """
    Perform t-tests to compare binary crossover types (1-point vs. 2-point), real crossover types (arithmetic vs. blx-alpha),
    and aggregated binary vs. real encodings.
    Parameters:
        df: DataFrame with GA results
        function_name: 'Ackley' or 'Rastrigin'
    Returns: Dict with t-test results for binary crossover comparison, real crossover comparison, and aggregated binary vs. real
    """
    crossovers = {'binary': ['1-point', '2-point'], 'real': ['arithmetic', 'blx-alpha']}
    df_func = df[df['Function'] == function_name]

    results = {
        'binary_crossover_comparison': {},
        'real_crossover_comparison': {},
        'aggregated_binary_vs_real': None
    }

    # Binary crossover comparison (1-point vs. 2-point)
    one_point_fitness = df_func[(df_func['Encoding'] == 'binary') & (df_func['Crossover'] == '1-point')]['Best_Fitness']
    two_point_fitness = df_func[(df_func['Encoding'] == 'binary') & (df_func['Crossover'] == '2-point')]['Best_Fitness']
    results['binary_crossover_comparison']['1-point_vs_2-point'] = perform_ttest(
        one_point_fitness, two_point_fitness, 'one_point', 'two_point', '1-point_vs_2-point'
    )

    # Real crossover comparison (arithmetic vs. blx-alpha)
    arithmetic_fitness = df_func[(df_func['Encoding'] == 'real') & (df_func['Crossover'] == 'arithmetic')]['Best_Fitness']
    blx_alpha_fitness = df_func[(df_func['Encoding'] == 'real') & (df_func['Crossover'] == 'blx-alpha')]['Best_Fitness']
    results['real_crossover_comparison']['arithmetic_vs_blx-alpha'] = perform_ttest(
        arithmetic_fitness, blx_alpha_fitness, 'arithmetic', 'blx_alpha', 'arithmetic_vs_blx-alpha'
    )

    # Aggregated comparison: binary (all crossovers) vs. real (all crossovers)
    binary_fitness = df_func[df_func['Encoding'] == 'binary']['Best_Fitness']
    real_fitness = df_func[df_func['Encoding'] == 'real']['Best_Fitness']
    results['aggregated_binary_vs_real'] = perform_ttest(
        binary_fitness, real_fitness, 'binary', 'real', 'binary_vs_real'
    )

    return results