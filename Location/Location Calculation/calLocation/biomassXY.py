import os
import pandas as pd

# 指定结果文件夹和output_csv文件夹的路径
ROOT=r"E:\Suyingcai\STV_MNet"
results_folder =ROOT+ r'\results\Structure calculation\results0.1'
output_csv_folder =ROOT+ r'\results\Location calculation\output_csv'

# 遍历结果文件夹下的所有{name}_result.csv文件
for csv_file in os.listdir(results_folder):
    if csv_file.endswith('_result.csv'):
        # 提取文件名（去除后缀）
        name = os.path.splitext(csv_file)[0].rsplit("_",1)[0]

        # 构建相应的output_csv文件名
        output_csv_file = f"pro_{name}.csv"
        result_csv_path = os.path.join(results_folder, csv_file)
        output_csv_path = os.path.join(output_csv_folder, output_csv_file)

        # 检查相应的output_csv文件是否存在
        if os.path.exists(output_csv_path):
            # 读取结果文件和output_csv文件
            result_df = pd.read_csv(result_csv_path)
            output_df = pd.read_csv(output_csv_path)

            # 将结果文件中的每一行数据的left、top、right、bottom对应的x、y数据从output_csv文件中找到并追加到结构测算结果文件中
            for index, row in result_df.iterrows():
                matched_row = output_df[output_df['ID'] == row['ID.1']]
                if not matched_row.empty:
                    result_df.loc[index, 'depth'] = matched_row['depth'].values[0]
                    # result_df.loc[index, 'y'] = matched_row['y'].values[0]
                    print("match",index)
                else:
                    print("dis_match",name,index)

            # 将更新后的结果写入到{name}_result.csv文件中
            result_df.to_csv(result_csv_path, index=False)

        else:
            print(f"Output CSV file {output_csv_file} not found for result CSV file {csv_file}")
print("done")
