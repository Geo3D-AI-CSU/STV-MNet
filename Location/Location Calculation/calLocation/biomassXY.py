import os
import pandas as pd

# Specify the paths for the results folder and output_csv folder
ROOT = r"E:\Suyingcai\STV_MNet"
results_folder = ROOT + r'\results\Structure calculation\results0.1'
output_csv_folder = ROOT + r'\results\Location calculation\output_csv'

# Iterate over all {name}_result.csv files in the results folder
for csv_file in os.listdir(results_folder):
    if csv_file.endswith('_result.csv'):
        # Extract the file name (without extension)
        name = os.path.splitext(csv_file)[0].rsplit("_", 1)[0]

        # Construct the corresponding output_csv file name
        output_csv_file = f"pro_{name}.csv"
        result_csv_path = os.path.join(results_folder, csv_file)
        output_csv_path = os.path.join(output_csv_folder, output_csv_file)

        # Check if the corresponding output_csv file exists
        if os.path.exists(output_csv_path):
            # Read the result file and output_csv file
            result_df = pd.read_csv(result_csv_path)
            output_df = pd.read_csv(output_csv_path)

            # For each row in the result file, find the corresponding x, y data from the output_csv file using left, top, right, bottom coordinates
            for index, row in result_df.iterrows():
                matched_row = output_df[output_df['ID'] == row['ID.1']]
                if not matched_row.empty:
                    result_df.loc[index, 'depth'] = matched_row['depth'].values[0]
                    # result_df.loc[index, 'y'] = matched_row['y'].values[0]
                    print("match", index)
                else:
                    print("dis_match", name, index)

            # Write the updated result to the {name}_result.csv file
            result_df.to_csv(result_csv_path, index=False)

        else:
            print(f"Output CSV file {output_csv_file} not found for result CSV file {csv_file}")
print("done")

