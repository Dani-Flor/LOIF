import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

############# Plot Test Results ###########################
# This function creates a line plot comparing F1-Score vs. The number of OTLs used for different OTL selection methods
# It gets the average for each method and uses 95% confidence intervals
# Parametes
# system: Name of the case system being used (Ex. case118,case_ieee30,etc.)
# test_results: This is a dataframe storing the results of each test (Ex. Test1 through Test24)
def plot_data(system, test_results,matrix,fc_otls, style=1):    
    # 1. Prepare Dataframe: Reset index (Methods) and map to full names
    summary = test_results.reset_index().rename(columns={"index": "Method"})
    method_map = {"HE": "High Eta", "Rand": "Random", "MCP": "MCP"}
    summary["Method"] = summary["Method"].map(method_map)
    
    # 2. Identify OTL columns (mix of ints and strings) and Melt
    otl_cols = [1, 2, 4, 8, 'FC'] 
    summary_long = summary.melt(
        id_vars=["Method", "DL"],
        value_vars=otl_cols,
        var_name="OTL",
        value_name="F1-score"
    )

    # 3. Categorical Order: Ensure "FC" stays at the end and types match
    otl_order = ["10", "20", "40", "80", "FC"]
    summary_long["OTL"] = summary_long["OTL"].astype(str)
    summary_long["OTL"] = pd.Categorical(summary_long["OTL"], categories=otl_order, ordered=True)


    if style == 1:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=summary_long, x="OTL", y="F1-score", hue="Method", 
                     errorbar=("ci", 95), marker="o")
        ax = plt.gca() # Get current axes
        ax.set_ylim(0, 1)
        # Set ticks to appear every 0.1 units
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

        # Optional: Add smaller "minor" ticks at 0.05 without labels
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        custom_x_labels = ['1','2','4','8',f'FC({len(fc_otls)})']
        ax.set_xticklabels(custom_x_labels)
        plt.ylabel("F1-score")
        plt.xlabel("OTL Level")
        plt.title(f"{system} – Method Performance With 95% CI ({matrix})")
        plt.grid(True, axis="y")
        plt.legend(title="Method")
        plt.tight_layout()
        plt.show()

