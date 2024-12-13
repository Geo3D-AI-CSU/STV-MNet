import os
import json
import csv


ROOT=r"E:\Suyingcai\STV_MNet"
input_csv_folder=ROOT+r"\data\input data\Location\LNinput_csv"

def json_to_csv(json_file_path, csv_file_path):
    # read JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # get shapes
    shapes = data['shapes']

    # write to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['label', 'left', 'right', 'top', 'bottom']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for shape in shapes:
            label = shape['label']
            left = shape['points'][0][0]
            right = shape['points'][1][0]
            top = shape['points'][2][1]
            bottom = shape['points'][1][1]
            writer.writerow({'label': label, 'left': left, 'right': right, 'top': top, 'bottom': bottom})

# all json data in folder
json_folder_path = ROOT+r"\data\input data\Location\LNbbox"
json_files = os.listdir(json_folder_path)

for json_file in json_files:
    if json_file.endswith(".json"):
        json_file_path = os.path.join(json_folder_path, json_file)
        csv_file_path = os.path.join(input_csv_folder, os.path.splitext(json_file)[0] + ".csv")
        json_to_csv(json_file_path, csv_file_path)
        print("done")

print("all done!")

