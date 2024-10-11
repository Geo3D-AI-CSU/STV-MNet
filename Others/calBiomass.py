
#计算文件的生物量、碳储量

import os
import pandas as pd

# 定义计算生物量的函数
def calculate_biomass(row):
    D = row['Tree_DBH']  # 树径
    H = row['Tree_H']    # 树高
    # 计算生物量的各部分
    W_S = 0.044 * ((D*100+0.83)**2 * H)**0.9169
    W_P = 0.023 * ((D*100+0.83)**2 * H)**0.7115
    W_B = 0.0104 * ((D*100+0.83)**2 * H)**0.9994
    W_L = 0.0188 * ((D*100+0.83)**2 * H)**0.8024
    W_R = 0.0197 * ((D*100+0.83)**2 * H)**0.8963

    # W_S = 0.044 * ((D*100)**2 * H)**0.9169
    # W_P = 0.023 * ((D*100)**2 * H)**0.7115
    # W_B = 0.0104 * ((D*100)**2 * H)**0.9994
    # W_L = 0.0188 * ((D*100)**2 * H)**0.8024
    # W_R = 0.0197 * ((D*100)**2 * H)**0.8963
    
    # 总生物量
    W_T = W_S + W_P + W_B + W_L + W_R

    
    return W_T/1000#单位t

# 指定结果文件夹路径
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# # 定义含碳率系数
carbon_coefficient = 0.48

file_path = os.path.join(results_folder, "results_Carbon_Clip_with_median.csv")
out_file_path=os.path.join(results_folder, "results_Carbon_Clip_with_median_i+1.csv")
# # # 读取CSV文件
df = pd.read_csv(file_path,encoding='utf-8-sig')

#计算生物量并追加到新列"Biomass"
df['Biomassi+1'] = df.apply(calculate_biomass, axis=1)
# print(df['Biomass'])
# 将"Biomass"列乘以含碳率系数，并添加到新列"Carbon"
df['Carboni+1'] = df['Biomassi+1'] * carbon_coefficient
# df_new=str(df)
#将结果写入原始文件
df.to_csv(file_path, index=False,encoding='utf-8-sig')
print("done")




#根据点空间连接结果字段JOIN_ID计算点缓冲区、道路总碳储量
import pandas as pd

# 读取CSV文件
df = pd.read_csv(results_folder +r"\线分割点C_exchange连接road_Split5000.csv",encoding='utf-8-sig')
# 根据group_field分组，计算sum_field的总和

# by_group_Carbon=df.groupby('JOIN_FID')['Carboni_1']
# by_group_Biomass=df.groupby('JOIN_FID')['Biomassi_1']

# sumBiomass_by_group = by_group_Biomass.sum().reset_index()
# sumCarbon_by_group = by_group_Carbon.sum().reset_index()

# 根据group_field分组，计算sum_field的总和
# sumBiomass_by_group= df.groupby('JOIN_FID')['Biomassi+1'].sum().reset_index()
# sumCarbon_by_group = df.groupby('JOIN_FID')['Carboni+1'].sum().reset_index()
sumCarbon_exchange_by_group = df.groupby('JOIN_FID')['Carbon_exchange'].sum().reset_index()


# # 将结果保存为新的CSV文件
# sumBiomass_by_group.to_csv(results_folder +r"\sumBiomassi+1_by_groupJOIN_FID.csv", index=False,encoding='utf-8-sig')
# sumCarbon_by_group.to_csv(results_folder +r"\sumCarboni+1_by_groupJOIN_FID.csv", index=False,encoding='utf-8-sig')


# # 将结果保存为新的CSV文件
# sumBiomass_by_group.to_csv(results_folder +r"\sumBiomassi+1_by_groupNEAR_ID.csv", index=False,encoding='utf-8-sig')
# sumCarbon_by_group.to_csv(results_folder +r"\sumCarboni+1_by_groupNEAR_ID.csv", index=False,encoding='utf-8-sig')
sumCarbon_exchange_by_group.to_csv(results_folder +r"\road_Split5000sumC_exchange.csv", index=False,encoding='utf-8-sig')






##根据['Carboni+1'] 和['Carbon']计算年碳汇量
# import pandas as pd

# # 读取CSV文件
# # df1 = pd.read_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final.csv")
# df2 = pd.read_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final_i+1.csv",encoding='utf-8-sig')

# # 将新列添加到原始文件中
# df2['Carbon_exchange'] = df2['Carboni+1'] - df2['Carbon']

# # 保存回原始文件
# df2.to_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final_i+1.csv", index=False,encoding='utf-8-sig')

