import pandas as pd
import os

# List to store all dataframes
all_dfs = []

catalogue_dir = 'stacked_catalogues'
for file in os.listdir(catalogue_dir):
    if file.startswith('CATALOGUE_') and file.endswith('.csv'):
        # Read the CSV from the catalogue directory
        df = pd.read_csv(os.path.join(catalogue_dir, file))

        # Extract method name from filename (removes 'CATALOGUE_' and '.csv')
        method = file.replace('CATALOGUE_', '').replace('.csv', '')

        # Add a column for the method
        df['method'] = method

        # Add to our list of dataframes
        all_dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save to Excel in the main directory
combined_df.to_excel('stacked_catalogues/Combined_DoR_Catalogues.xlsx', index=False)

print(f"Combined {len(all_dfs)} catalogues into 'Combined_DoR_Catalogues.xlsx'")