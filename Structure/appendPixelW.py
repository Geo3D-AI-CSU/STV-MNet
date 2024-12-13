import os
import pandas as pd
import glob

# Define the root directory
ROOT = "E:\Suyingcai\STV_MNet"
input_dir = ROOT + r"\results\Structure calculation\results0.1"

# List all files in the input directory
input_fileNames = os.listdir(input_dir)

for fileName in input_fileNames:
    # Construct the full file path
    file = os.path.join(input_dir, fileName)
    # Read the summary CSV file
    df_summary = pd.read_csv(file, encoding='utf-8-sig')

    # Iterate through each row of the summary file
    for index, row in df_summary.iterrows():
        if pd.notnull(row['ID']):  # If ID is not null
            # Construct the path for the corresponding CSV file in the 'results_Wr' folder
            csv_file = os.path.join(ROOT + r"\results\Structure calculation\results_Wr", fileName)
            
            # Check if the file exists
            if os.path.exists(csv_file):
                
                # Read the corresponding CSV file
                df_csv = pd.read_csv(csv_file, encoding='utf-8-sig')
                
                # Find data with the same ID
                matching_row = df_csv[df_csv['ID'] == row['ID']]
                if not matching_row.empty:
                    # Append the PixelWidth data to the summary file
                    pixel_width = matching_row.iloc[0]['PixelWidth']
                    df_summary.at[index, 'PixelWidth'] = pixel_width
            else:
                # Print the path if the file does not exist
                print(csv_file)
    # print(df_summary)

    # Save the updated summary file
    df_summary.to_csv(file, index=False, encoding='utf-8-sig')
