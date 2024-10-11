import pandas as pd
import numpy as np
from scipy.interpolate import griddata
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# 读取CSV文件
file_path = results_folder +r'\线分割点连接C_exchange.csv'
df = pd.read_csv(file_path)

# 提取非空Carbon的点
known_points = df.dropna(subset=['Carbon_exchange'])

# 提取已知点的坐标和Carbon值
points = known_points[['POINT_X', 'POINT_Y']].values
values = known_points['Carbon_exchange'].values

# 提取待插值的点的坐标
unknown_points = df[df['Carbon_exchange'].isna()][['POINT_X', 'POINT_Y']].values

# 进行插值（先使用线性插值）
interpolated_values = griddata(points, values, unknown_points, method='linear')

# 检查是否有未插值的点
missing_values_mask = np.isnan(interpolated_values)
if np.any(missing_values_mask):
    # 对未插值的点使用最近邻插值
    interpolated_values[missing_values_mask] = griddata(points, values, unknown_points[missing_values_mask], method='nearest')

# 将插值结果赋值回原数据框
df.loc[df['Carbon_exchange'].isna(), 'Carbon_exchange'] = interpolated_values

# 保存结果到新的CSV文件
output_file_path = results_folder +r'\插值结果C_exchange.csv'
df.to_csv(output_file_path, index=False)

print(f"插值结果已保存至 {output_file_path}")
