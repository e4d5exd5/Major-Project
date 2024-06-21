import pandas as pd
import numpy as np
import os

def load_csv(csv_path):
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    return df


def load_total_csv(folder_path):

    newdf = pd.DataFrame(columns=['dataset', 'encoder', 'tau', 'n_times', 'run','OA', 'KA', 'AA'])
    ds_folders = os.listdir(folder_path)
    for ds_folder in ds_folders:
        dataset = ds_folder
        if not os.path.isdir(folder_path + ds_folder):
            continue
        enc_folders = os.listdir(folder_path + ds_folder)
        for enc_folder in enc_folders:
            encoder = enc_folder
            if not os.path.isdir(folder_path + ds_folder + '/' + enc_folder):
                continue
            tau_folders = os.listdir(folder_path + ds_folder + '/' + enc_folder)
            tau_folders.sort(key=float)
            for tau_folder in tau_folders:
                tau = tau_folder
                if not os.path.isdir(folder_path + ds_folder + '/' + enc_folder + '/' + tau_folder):
                    continue
                folders = os.listdir(folder_path + ds_folder + '/' + enc_folder + '/' +tau_folder)    
                folders.sort(key=int)
                for folder in folders:
                    n_times = int(folder)
                    if os.path.isdir(folder_path + ds_folder + '/' + enc_folder + '/' + tau_folder + '/' + folder):
                        files = os.listdir(folder_path + ds_folder + '/' + enc_folder + '/' + tau_folder + '/' + folder)
                        for file in files:
                            if file.endswith('_post_tune_reportTotal.txt'):
                                file_path = folder_path + ds_folder + '/' + enc_folder + '/' + tau_folder + '/' + folder + '/' + file
                                run = int(file.split('_')[0])
                                with open(file_path, 'r') as f:
                                    lines = f.readlines()
                                    OA = float(lines[3].split()[0])
                                    KA = float(lines[4].split()[0])
                                    AA = float(lines[5].split()[0])
                                    newdf.loc[len(newdf.index)] = [dataset, encoder, tau, n_times, run, OA, KA, AA]
    
    # print(df)
    return newdf
                        
                        

def process_csv(df):
    

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



def main():
    # Example usage:
    table = './ablation_review/tr 40 tu 15 1 shot ksc/'
    csv_path = table + 'report.csv'
    folder_path = table + 'IP/conv_sa/1/'
    datasets = ['IP']
    taus = ['1', '1.8']
    encoders = ['conv_no_sa']

    input_df = load_csv(csv_path)
    input_df.columns = ['dataset','encoder','tau','n_times','run','OA','KA','AA']
    print("Actual Testing Result")
    result_actual_df = process_csv(input_df)
    print(result_actual_df)
    print("\n")
    with open(table + 'calculation.txt', '+a') as file:
        file.write('Actual Testing Results: \n ')
        file.write(result_actual_df.to_string(header=True, index=True))
        
    input_total_df = load_total_csv(table)
    print("Post Tune Visualization Result:")
    result_post_df = process_csv(input_total_df)
    print(result_post_df)
    with open(table + 'calculation.txt', '+a') as file:
        file.write('\n\nPost Tune Visualization Result: \n ')
        file.write(result_post_df.to_string(header=True, index=True))
        
                
    # Print the result DataFrame
    # print(result_df)
    
    
if __name__ == '__main__':
    main()

