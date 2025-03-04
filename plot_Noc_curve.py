import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_Noc(csv_path):
    df = pd.read_csv(csv_path, index_col='Iteration')

    thresholds = [0.8, 0.85, 0.90]

    results = {threshold: [] for threshold in thresholds}

    for column in df.columns:
        for threshold in thresholds:
            filtered_df = df[column][df[column] > threshold]
            if not filtered_df.empty:
                results[threshold].append(filtered_df.index[0])
            else:
                results[threshold].append(20)
    averages = {threshold: np.mean(results[threshold]) for threshold in thresholds}

    print("Averages for each threshold:", averages)

    mean_values = df.iloc[:20].mean(axis=1)
    print("Mean values for the first 20 rows:", mean_values)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mean_values) + 1), mean_values, marker='o', linestyle='-', color='b')
    plt.title('Mean Value Curve for First 20 Rows')
    plt.xlabel('Row Index')
    plt.ylabel('Mean Value')
    plt.xticks(range(1, len(mean_values) + 1))
    plt.grid(True)
    plt.show()


plot_Noc("results.csv")
