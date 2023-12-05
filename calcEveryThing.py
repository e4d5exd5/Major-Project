import pandas as pd
import numpy as np


def process_csv(csv_path):
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Find the row with the maximum 'OA', 'KA', and 'AA' with priority
    result_df = df.loc[df.groupby(['dataset', 'encoder', 'tau', 'n_times'])[
        'OA'].idxmax()]

    # If there's a tie in 'OA', use 'KA'
    result_df = result_df.loc[result_df.groupby(
        ['dataset', 'encoder', 'tau', 'n_times'])['KA'].idxmax()]

    # If there's a tie in both 'OA' and 'KA', use 'AA'
    result_df = result_df.loc[result_df.groupby(
        ['dataset', 'encoder', 'tau', 'n_times'])['AA'].idxmax()]

    # Calculate mean and variance for 'OA', 'KA', and 'AA'
    mean_var_df = df.groupby(['dataset', 'encoder', 'tau', 'n_times']).agg({
        'OA': ["mean", "var"],
        'KA': ["mean", "var"],
        'AA': ["mean", "var"]
    }).reset_index()
    mean_var_df.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in mean_var_df.columns]

    
    # Merge the result and mean/variance DataFrames
    result_df = pd.merge(result_df, mean_var_df, on=[
                         'dataset', 'encoder', 'tau', 'n_times'], suffixes=('_max', '_mean_var'))

    # Drop the 'run' column
    result_df = result_df.drop(columns=['run'])

    return result_df


# Example usage:
csv_path = './ablation/table test/report.csv'
result_df = process_csv(csv_path)

# Print the result DataFrame
print(result_df)
