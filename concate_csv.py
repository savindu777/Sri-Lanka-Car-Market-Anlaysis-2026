import glob
import pandas as pd

# 1. Find all CSV files inside the 'dataset' folder
all_csv_files = glob.glob('dataset/*.csv')

print(f"Found files to concatenate: {all_csv_files}")

# 2. Read and combine all CSV files into a list of DataFrames
df_list = [pd.read_csv(file) for file in all_csv_files]

# Concatenate all DataFrames (automatically aligns columns across different models)
combined_df = pd.concat(df_list, ignore_index=True)

# 3. Save the combined DataFrame to a new CSV file
output_file = 'combined_riyasewana_datasets.csv'
combined_df.to_csv(output_file, index=False)

print(f"Successfully concatenated {len(all_csv_files)} files into '{output_file}'!")
print(f"Total rows in combined dataset: {len(combined_df)}")