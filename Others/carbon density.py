import pandas as pd
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# read CSV 
df = pd.read_csv(results_folder +r"\Carbon_density.csv",encoding='utf-8-sig')

df['Density'] = df["Carbon"]/df['Shape_Leng']

df.to_csv(results_folder +r"\Carbon_density.csv", index=False,encoding='utf-8-sig')

print("统计结果已保存。")
