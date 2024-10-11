import os
import pandas as pd
import glob
# 读取汇总文件
ROOT="E:\Suyingcai\STV_MNet"
input_dir =ROOT+r"\results\Structure calculation\results0.1"
input_fileNames=os.listdir(input_dir )
for fileName in input_fileNames:
    file=os.path.join(input_dir,fileName)
    df_summary = pd.read_csv(file,encoding='utf-8-sig')

    # 遍历汇总文件中的每一行
    for index, row in df_summary.iterrows():
        if pd.notnull(row['ID']):  # 如果ID不为空
            csv_file =os.path.join(ROOT+r"\results\Structure calculation\results_Wr",fileName) 
            
            # 检查文件是否存在

            if os.path.exists(csv_file):
                
                df_csv = pd.read_csv(csv_file,encoding='utf-8-sig')
                
                # 找到具有相同ID的数据
                matching_row = df_csv[df_csv['ID'] == row['ID']]
                if not matching_row.empty:
                    # 将PixelWidth数据追加到汇总文件中
                    pixel_width = matching_row.iloc[0]['PixelWidth']
                    df_summary.at[index, 'PixelWidth'] = pixel_width
            else:
                print(csv_file)
    # print(df_summary)

    # 保存更新后的汇总文件
    df_summary.to_csv(file, index=False,encoding='utf-8-sig')
