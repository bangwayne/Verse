import pandas as pd
import numpy as np

def calculate_Dice(csv_path, mode=3):
    df = pd.read_csv(csv_path, index_col='Iteration')
    assert mode in [2, 3], "Mode should be 2 or 3"
    thresholds = [0.85, 0.90, 0.95]
    results = {threshold: [] for threshold in thresholds}
    for column in df.columns:
        for threshold in thresholds:
            filtered_df = df[column][df[column] > threshold]
            if not filtered_df.empty:
                if mode == 2:
                    results[threshold].append(filtered_df.index[0])
                elif mode == 3:
                    results[threshold].append(filtered_df.index[0] + 1)
            else:
                results[threshold].append(20)

    averages = {threshold: np.mean(results[threshold]) for threshold in thresholds}
    print("Averages for each threshold:", averages)
    mean_first_row = df.iloc[0].mean()
    print(len(df.iloc[0]))
    mean_five_row = df.iloc[4].mean()
    print(f"Mean of 5 click row: {mean_five_row}")
    mean_ten_row = df.iloc[9].mean()
    print(f"Mean of 10 click row: {mean_ten_row}")
    if len(df) > 20:
        if mode == 2:
            mean_twentieth_row = df.iloc[20].mean()
        elif mode == 3:
            mean_twentieth_row = df.iloc[19].mean()
    else:
        mean_twentieth_row = None  # or appropriate handling if the row does not exist
    if mode == 2:
        print(f"Mean of 0 click row: {mean_first_row}")
        print(f"Mean of 1 click row: {df.iloc[1].mean()}")
    elif mode == 3:
        print(f"Mean of 1 click row: {mean_first_row}")

    if mean_twentieth_row is not None:
        print(f"Mean of 20 click row: {mean_twentieth_row}")
    else:
        print("Twentieth row does not exist.")


root_file = "/research/cbim/vast/bg654/Desktop/jupyproject/Verse_git"
calculate_Dice(root_file + '/Verse_Cardiac_mode3_dice_ACDC_results.csv')
