import os
import shutil

# 指定结果文件夹和bbox文件夹的路径
ROOT=r"E:\Suyingcai\STV_MNet"
results_folder =ROOT+ r'\results\Structure calculation\results0.1'
bbox_folder = ROOT+ r'\data\input data\Structure\bbox'
LNjson_folder =ROOT+  r'\data\input data\Location\LNbbox'

# 确保LNjson文件夹存在，如果不存在则创建
if not os.path.exists(LNjson_folder):
    os.makedirs(LNjson_folder)

# 遍历结果文件夹下的所有CSV文件
for csv_file in os.listdir(results_folder):
    if csv_file.endswith('_result.csv'):
        # 提取文件名（去除后缀）
        name = os.path.splitext(csv_file)[0]

        # 构建对应的JSON文件名
        json_file = name.rsplit("_",1)[0] + '.json'

        # 检查对应的JSON文件是否存在于bbox文件夹下
        if os.path.exists(os.path.join(bbox_folder, json_file)):
            # 如果存在，则复制到LNjson文件夹下
            shutil.copy(os.path.join(bbox_folder, json_file), os.path.join(LNjson_folder, json_file))
        else:
            print(f"JSON file {json_file} not found for CSV file {csv_file}")
