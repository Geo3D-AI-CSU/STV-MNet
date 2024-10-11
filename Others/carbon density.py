import pandas as pd
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# 读取 CSV 文件
df = pd.read_csv(results_folder +r"\Carbon_density.csv",encoding='utf-8-sig')

# 根据 JOIN_FID 分组，计算 Carbon 列的和
df['Density'] = df["Carbon"]/df['Shape_Leng']

# 将结果保存为新表
df.to_csv(results_folder +r"\Carbon_density.csv", index=False,encoding='utf-8-sig')

print("统计结果已保存。")
