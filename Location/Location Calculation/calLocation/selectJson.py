import os
import shutil

# Specify the paths for the results folder and the bbox folder
ROOT = r"E:\Suyingcai\STV_MNet"
results_folder = ROOT + r'\results\Structure calculation\results0.1'
bbox_folder = ROOT + r'\data\input data\Structure\bbox'
LNjson_folder = ROOT + r'\data\input data\Location\LNbbox'

# Ensure the LNjson folder exists, create it if it doesn't
if not os.path.exists(LNjson_folder):
    os.makedirs(LNjson_folder)

# Iterate through all CSV files in the results folder
for csv_file in os.listdir(results_folder):
    if csv_file.endswith('_result.csv'):
        # Extract the file name (remove the extension)
        name = os.path.splitext(csv_file)[0]

        # Construct the corresponding JSON file name
        json_file = name.rsplit("_", 1)[0] + '.json'

        # Check if the corresponding JSON file exists in the bbox folder
        if os.path.exists(os.path.join(bbox_folder, json_file)):
            # If it exists, copy it to the LNjson folder
            shutil.copy(os.path.join(bbox_folder, json_file), os.path.join(LNjson_folder, json_file))
        else:
            print(f"JSON file {json_file} not found for CSV file {csv_file}")
