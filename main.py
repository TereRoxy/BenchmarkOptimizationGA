import tkinter as tk
from functions_plots import ackley, rastrigin, plot_function
from gui import GAInterface
from analysis import compute_summary, plot_boxplot_to_file, ttest_compare_encodings, load_results

def main():
    # Generate function plots
    plot_function(ackley, 'Ackley', (-10, 10), (-10, 10), 'ackley')
    plot_function(rastrigin, 'Rastrigin', (-5, 5), (-5, 5), 'rastrigin')

    # Load GA results
    results_df = load_results()

    # Compute summary statistics
    summary_df = compute_summary(results_df)
    print("Summary Statistics:")
    print(summary_df)

    # Generate box plots for both functions
    plot_boxplot_to_file(results_df, 'Ackley', 'ackley_boxplot.png')
    plot_boxplot_to_file(results_df, 'Rastrigin', 'rastrigin_boxplot.png')

    # Perform t-tests for both functions
    for function in ['Ackley', 'Rastrigin']:
        print(f"\nT-test Results for {function}:")
        results = ttest_compare_encodings(results_df, function)

        # Binary crossover comparison
        print("Binary Crossover Comparison (1-point vs. 2-point):")
        binary_comp = results['binary_crossover_comparison']['1-point_vs_2-point']
        if binary_comp['t_stat'] is not None:
            print(f"1-point vs. 2-point: t={binary_comp['t_stat']:.4f}, p={binary_comp['p_value']:.4f}, "
                  f"Significant={binary_comp['significant']}, "
                  f"1-point Mean={binary_comp['mean_one_point']:.4e}, "
                  f"2-point Mean={binary_comp['mean_two_point']:.4e}, "
                  f"Samples={binary_comp['sample_size_one_point']}/{binary_comp['sample_size_two_point']}")
        else:
            print(f"1-point vs. 2-point: {binary_comp['error']}, "
                  f"1-point Mean={binary_comp['mean_one_point']:.4e}, "
                  f"2-point Mean={binary_comp['mean_two_point']:.4e}, "
                  f"Samples={binary_comp['sample_size_one_point']}/{binary_comp['sample_size_two_point']}")

        # Real crossover comparison
        print("\nReal Crossover Comparison (arithmetic vs. blx-alpha):")
        real_comp = results['real_crossover_comparison']['arithmetic_vs_blx-alpha']
        if real_comp['t_stat'] is not None:
            print(f"arithmetic vs. blx-alpha: t={real_comp['t_stat']:.4f}, p={real_comp['p_value']:.4f}, "
                  f"Significant={real_comp['significant']}, "
                  f"arithmetic Mean={real_comp['mean_arithmetic']:.4e}, "
                  f"blx-alpha Mean={real_comp['mean_blx_alpha']:.4e}, "
                  f"Samples={real_comp['sample_size_arithmetic']}/{real_comp['sample_size_blx_alpha']}")
        else:
            print(f"arithmetic vs. blx-alpha: {real_comp['error']}, "
                  f"arithmetic Mean={real_comp['mean_arithmetic']:.4e}, "
                  f"blx-alpha Mean={real_comp['mean_blx_alpha']:.4e}, "
                  f"Samples={real_comp['sample_size_arithmetic']}/{real_comp['sample_size_blx_alpha']}")


        # Aggregated binary vs. real
        print("\nAggregated Binary vs. Real:")
        agg = results['aggregated_binary_vs_real']
        if agg['t_stat'] is not None:
            print(f"Aggregated: t={agg['t_stat']:.4f}, p={agg['p_value']:.4f}, "
                  f"Significant={agg['significant']}, "
                  f"Binary Mean={agg['mean_binary']:.4e}, "
                  f"Real Mean={agg['mean_real']:.4e}, "
                  f"Samples={agg['sample_size_binary']}/{agg['sample_size_real']}")
        else:
            print(f"Aggregated: {agg['error']}, "
                  f"Binary Mean={agg['mean_binary']:.4e}, "
                  f"Real Mean={agg['mean_real']:.4e}, "
                  f"Samples={agg['sample_size_binary']}/{agg['sample_size_real']}")

    # Run GUI
    root = tk.Tk()
    app = GAInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()