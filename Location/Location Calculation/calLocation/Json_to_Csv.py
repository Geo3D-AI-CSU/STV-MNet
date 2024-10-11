import os
import json
import csv


ROOT=r"E:\Suyingcai\STV_MNet"
input_csv_folder=ROOT+r"\data\input data\Location\LNinput_csv"

# 定义json_to_csv函数
def json_to_csv(json_file_path, csv_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 提取shapes数据
    shapes = data['shapes']

    # 写入CSV文件
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

# 获取文件夹中的所有JSON文件
json_folder_path = ROOT+r"\data\input data\Location\LNbbox"
json_files = os.listdir(json_folder_path)

# 对每个JSON文件运行json_to_csv函数
for json_file in json_files:
    if json_file.endswith(".json"):
        json_file_path = os.path.join(json_folder_path, json_file)
        csv_file_path = os.path.join(input_csv_folder, os.path.splitext(json_file)[0] + ".csv")
        json_to_csv(json_file_path, csv_file_path)
        print("done")

print("all done!")

