import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

# 定义每次处理的批次大小
batch_size = 2000
input_file = r"C:\data\new_structure\test\final_utm.csv"
output_file = r"C:\data\new_structure\test\test_output_275_del.csv"

# 定义初始的DataFrame为空
final_results = pd.DataFrame()

# 获取CSV文件的行数
total_rows = sum(1 for line in open(input_file)) - 1  # 减去表头行

k=1

# 分批次读取文件
for start_row in range(0, total_rows, batch_size):
    # 读取当前批次的数据
    df_batch = pd.read_csv(input_file, skiprows=range(1, start_row + 1), nrows=batch_size)

    # 获取坐标和其他数据
    coordinates = df_batch[['UTM_X', 'UTM_Y']].values
    heights = df_batch['Tree_H'].values
    diameters = df_batch['Tree_DBH'].values 

    # 计算距离矩阵
    distance_matrix = pairwise_distances(coordinates)

    # 进行层次聚类
    Z = linkage(distance_matrix, method='ward')

    # 根据阈值将数据分簇
    # 275对等实际距离3m
    threshold = 275
    clusters = fcluster(Z, threshold, criterion='distance')

    # 将聚类结果存入 DataFrame
    df_batch['cluster'] = clusters

    print("已处理"+str(k)+"个批次")
    k=k+1

    # 检查每个簇内的树高和胸径
    new_clusters = []
    for cluster in df_batch['cluster'].unique():
        cluster_data = df_batch[df_batch['cluster'] == cluster]
        
        # 计算树高和胸径的标准差
        height_std = cluster_data['Tree_H'].std()
        diameter_std = cluster_data['Tree_DBH'].std()
        
        # 设定标准差阈值
        height_threshold = 3
        diameter_threshold = 0.3
        
        if height_std > height_threshold and diameter_std > diameter_threshold:
            # 细分该簇
            for i, row in cluster_data.iterrows():
                new_clusters.append(i)
        else:
            for _ in range(len(cluster_data)):
                new_clusters.append(cluster)

    # 为不符合条件的树木赋予新簇编号
    if len(new_clusters) > len(df_batch['cluster'].unique()):
        new_cluster_num = max(clusters) + 1
        for i, idx in enumerate(new_clusters):
            if new_clusters[i] not in df_batch['cluster'].unique():
                df_batch.at[idx, 'cluster'] = new_cluster_num
                new_cluster_num += 1

    # 计算每个簇的平均坐标
    average_coordinates = df_batch.groupby('cluster').agg(
        x_mean=('UTM_X', 'mean'),
        y_mean=('UTM_Y', 'mean'),
        cluster_size=('cluster', 'size')
    ).reset_index()

    # 计算每个簇内树高和胸径的平均值
    height_diameter_means = df_batch.groupby('cluster').agg(
        tree_height_mean=('Tree_H', 'mean'), 
        tree_diameter_mean=('Tree_DBH', 'mean')
    ).reset_index()

    # 合并两个 DataFrame
    batch_results = average_coordinates.merge(height_diameter_means, on='cluster')

    # 过滤掉只有一个点的簇
    #batch_results = batch_results[['x_mean', 'y_mean', 'tree_height_mean', 'tree_diameter_mean']]
    batch_results = batch_results[batch_results['cluster_size'] > 1][['x_mean', 'y_mean', 'tree_height_mean', 'tree_diameter_mean']]

    # 将批次结果添加到最终结果中
    final_results = pd.concat([final_results, batch_results], ignore_index=True)

# 将最终结果保存到CSV文件
final_results.to_csv(output_file, index=False)

# 输出结果
print(f"处理完成，结果已保存至：{output_file}")
print(f"总行数: {len(final_results)}")