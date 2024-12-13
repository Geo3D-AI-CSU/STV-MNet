import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

# Define the batch size for each process
batch_size = 2000
input_file = r"C:\data\new_structure\test\final_utm.csv"
output_file = r"C:\data\new_structure\test\test_output_275_del.csv"

# Initialize an empty DataFrame for the final results
final_results = pd.DataFrame()

# Get the total number of rows in the CSV file
total_rows = sum(1 for line in open(input_file)) - 1  # Subtract the header row

k = 1

# Process the file in batches
for start_row in range(0, total_rows, batch_size):
    # Read the current batch of data
    df_batch = pd.read_csv(input_file, skiprows=range(1, start_row + 1), nrows=batch_size)

    # Extract coordinates and other data
    coordinates = df_batch[['UTM_X', 'UTM_Y']].values
    heights = df_batch['Tree_H'].values
    diameters = df_batch['Tree_DBH'].values 

    # Compute the distance matrix
    distance_matrix = pairwise_distances(coordinates)

    # Perform hierarchical clustering
    Z = linkage(distance_matrix, method='ward')

    # Cluster the data based on a threshold distance
    # 275 corresponds to an actual distance of 3 meters
    threshold = 275
    clusters = fcluster(Z, threshold, criterion='distance')

    # Store the clustering results in the DataFrame
    df_batch['cluster'] = clusters

    print(f"Processed batch {k}")
    k += 1

    # Check the tree heights and diameters within each cluster
    new_clusters = []
    for cluster in df_batch['cluster'].unique():
        cluster_data = df_batch[df_batch['cluster'] == cluster]
        
        # Calculate the standard deviation of tree height and diameter
        height_std = cluster_data['Tree_H'].std()
        diameter_std = cluster_data['Tree_DBH'].std()
        
        # Set thresholds for standard deviations
        height_threshold = 3
        diameter_threshold = 0.3
        
        if height_std > height_threshold and diameter_std > diameter_threshold:
            # Subdivide the cluster
            for i, row in cluster_data.iterrows():
                new_clusters.append(i)
        else:
            for _ in range(len(cluster_data)):
                new_clusters.append(cluster)

    # Assign new cluster IDs for trees that don't meet the criteria
    if len(new_clusters) > len(df_batch['cluster'].unique()):
        new_cluster_num = max(clusters) + 1
        for i, idx in enumerate(new_clusters):
            if new_clusters[i] not in df_batch['cluster'].unique():
                df_batch.at[idx, 'cluster'] = new_cluster_num
                new_cluster_num += 1

    # Calculate the average coordinates for each cluster
    average_coordinates = df_batch.groupby('cluster').agg(
        x_mean=('UTM_X', 'mean'),
        y_mean=('UTM_Y', 'mean'),
        cluster_size=('cluster', 'size')
    ).reset_index()

    # Calculate the average tree height and diameter for each cluster
    height_diameter_means = df_batch.groupby('cluster').agg(
        tree_height_mean=('Tree_H', 'mean'), 
        tree_diameter_mean=('Tree_DBH', 'mean')
    ).reset_index()

    # Merge the two DataFrames
    batch_results = average_coordinates.merge(height_diameter_means, on='cluster')

    # Filter out clusters with only one point
    # batch_results = batch_results[['x_mean', 'y_mean', 'tree_height_mean', 'tree_diameter_mean']]
    batch_results = batch_results[batch_results['cluster_size'] > 1][['x_mean', 'y_mean', 'tree_height_mean', 'tree_diameter_mean']]

    # Append the batch results to the final results
    final_results = pd.concat([final_results, batch_results], ignore_index=True)

# Save the final results to a CSV file
final_results.to_csv(output_file, index=False)

# Output the result
print(f"Processing complete, results have been saved to: {output_file}")
print(f"Total number of rows: {len(final_results)}")
