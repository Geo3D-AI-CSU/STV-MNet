# Calculate Carbon Emissions
# -*- coding: utf-8 -*-
# import pandas as pd
# results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# # Read CSV file
# df = pd.read_csv(results_folder +r"\碳排放量计算.csv", encoding="utf-8-sig")

# # Loop through each row to calculate carbon emissions and add the results to the DataFrame
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

# # Add the calculated results as a new column to the DataFrame
# df["Carbon_Emission"] = carbon_emissions
# print(df)
# # Save the updated DataFrame back to a CSV file
# df.to_csv(results_folder +r"\碳排放量计算.csv", index=False, encoding="utf-8-sig")
# print("OK!")

# Calculate Carbon Source-Sink Matching
import pandas as pd
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# Read CSV files
carbon_emission_df = pd.read_csv(results_folder +r'\碳排放量计算.csv',encoding='utf-8-sig')
carbon_exchange_df = pd.read_csv(results_folder +r'\sumCarbon_exchange_by_groupNEAR_ID.csv')

# Rename columns for matching
carbon_emission_df.rename(columns={'FID': 'NEAR_FID'}, inplace=True)

# Merge the two DataFrames based on FID and NEAR_FID
merged_df = carbon_exchange_df.merge(carbon_emission_df[['NEAR_FID', 'Carbon_Emission']], on='NEAR_FID', how='left')

# Calculate the C_match field
merged_df['C_match'] = merged_df['Carbon_Emission'] - merged_df['Carbon_exchange']

# Save the result to a new CSV file
merged_df.to_csv(results_folder +r'\sumC_match.csv', index=False,encoding='utf-8-sig')

print("Merging completed and saved as sumC_match.csv")
