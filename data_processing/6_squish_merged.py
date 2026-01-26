import pandas as pd
import glob

# 1. Find all csv files in the current folder
#    (Use 'path/to/folder/*.csv' if they aren't in the same folder)
file_list = glob.glob('4_merged_data/*.csv')

# 2. Read each file into a list of DataFrames
dfs = []
for file in file_list:
    df = pd.read_csv(file)
    dfs.append(df)

# 3. Stack them vertically
#    ignore_index=True ensures the new file has a fresh, continuous index
master_df = pd.concat(dfs, ignore_index=True)

# 4. Save to a new file
master_df.to_csv('5_weatherid_merged_output.csv', index=False)

print(f"Merged {len(file_list)} files")