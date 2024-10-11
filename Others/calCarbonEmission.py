

#计算碳排放量
# -*- coding: utf-8 -*-
# import pandas as pd
#results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# # 读取 CSV 文件
# df = pd.read_csv(results_folder +r"\碳排放量计算.csv", encoding="utf-8-sig")

# # 逐行计算碳排放量，并将结果添加到 DataFrame 中
# carbon_emissions = []
# for index, row in df.iterrows():
#     if row["ClassName"]=="高速公路":
#         N=2500
#     elif row["ClassName"]=="城市一级道路":
#         N=1500
#     elif row["ClassName"]=="城市二级道路":
#         N=3000
#     elif row["ClassName"]=="城市三级道路":
#         N=1000
#     else :
#         N=200
#     e = 0.35
#     n = 43.0
#     C = 20.0
#     O = 1
#     L = row["Shape_Leng"] * N * 365
#     F = L * e
#     P = n * F * C * O * (44 / 12)
#     carbon_emissions.append(P)

# # 将计算结果作为新列添加到 DataFrame 中
# df["Carbon_Emission"] = carbon_emissions
# print(df)
# # 将更新后的 DataFrame 保存回 CSV 文件
# df.to_csv(results_folder +r"\碳排放量计算.csv", index=False, encoding="utf-8-sig")
# print("OK!")


#计算碳源汇匹配
import pandas as pd
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# 读取CSV文件
carbon_emission_df = pd.read_csv(results_folder +r'\碳排放量计算.csv',encoding='utf-8-sig')
carbon_exchange_df = pd.read_csv(results_folder +r'\sumCarbon_exchange_by_groupNEAR_ID.csv')

# 重命名列以便于匹配
carbon_emission_df.rename(columns={'FID': 'NEAR_FID'}, inplace=True)

# 合并两个数据框以匹配FID和NEAR_FID
merged_df = carbon_exchange_df.merge(carbon_emission_df[['NEAR_FID', 'Carbon_Emission']], on='NEAR_FID', how='left')

# 计算C_match字段
merged_df['C_match'] = merged_df['Carbon_Emission'] - merged_df['Carbon_exchange']

# 保存结果到新的CSV文件
merged_df.to_csv(results_folder +r'\sumC_match.csv', index=False,encoding='utf-8-sig')

print("合并完成并已保存为sumC_match.csv")


