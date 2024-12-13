import pandas as pd
import os

# Get all the CSV files in the folder
ROOT = r"E:\Suyingcai\STV_MNet"
folder_path = ROOT + r"\results\Structure calculation\results0.1"
files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Create an empty DataFrame to store all the data
combined_data = pd.DataFrame()

# Iterate through all the CSV files and merge the data
for file in files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)  # Read CSV file data
    combined_data = combined_data.append(data, ignore_index=True)  # Merge data
    print(file)

# Save the merged data to a new CSV file
combined_data.to_csv(folder_path + r"\results_all.csv", index=False)

