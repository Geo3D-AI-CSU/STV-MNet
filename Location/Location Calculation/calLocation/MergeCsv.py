import pandas as pd
import os

# 获取文件夹中的所有csv文件
ROOT=r"E:\Suyingcai\STV_MNet"
folder_path = ROOT+r"\results\Structure calculation\results0.1"
files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 创建一个空的DataFrame来存储所有数据
combined_data = pd.DataFrame()

# 遍历所有的csv文件并合并数据
for file in files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)  # 读取csv文件数据
    combined_data = combined_data.append(data, ignore_index=True)  # 合并数据
    print(file)

# 将合并后的数据保存为新的csv文件
combined_data.to_csv(folder_path+r"\results_all.csv", index=False)
