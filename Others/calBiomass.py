import os
import pandas as pd

# Define a function to calculate biomass
def calculate_biomass(row):
    D = row['Tree_DBH']  # Tree Diameter at Breast Height
    H = row['Tree_H']    # Tree Height
    # Calculate different parts of biomass
    W_S = 0.044 * ((D*100+0.83)**2 * H)**0.9169 # stem biomass
    W_P = 0.023 * ((D*100+0.83)**2 * H)**0.7115 # branch biomass
    W_B = 0.0104 * ((D*100+0.83)**2 * H)**0.9994 # bark biomass
    W_L = 0.0188 * ((D*100+0.83)**2 * H)**0.8024 # leaf biomass
    W_R = 0.0197 * ((D*100+0.83)**2 * H)**0.8963 # root biomass
    
    # Total biomass
    W_T = W_S + W_P + W_B + W_L + W_R
    
    return W_T/1000# unit in tons

# Specify the results folder path
results_folder = r"E:\Suyingcai\STV_MNet\data\Carbon data"
# Define the carbon coefficient
carbon_coefficient = 0.48

file_path = os.path.join(results_folder, "results_Carbon_Clip_with_median.csv")
out_file_path=os.path.join(results_folder, "results_Carbon_Clip_with_median_i+1.csv")
# Read the CSV file
df = pd.read_csv(file_path,encoding='utf-8-sig')

# Calculate biomass and add it to a new column "Biomass"
df['Biomassi+1'] = df.apply(calculate_biomass, axis=1)
# Multiply "Biomass" column by the carbon coefficient and add it to a new column "Carbon"
df['Carboni+1'] = df['Biomassi+1'] * carbon_coefficient
# df_new=str(df)
# Write the results back to the original file
df.to_csv(file_path, index=False,encoding='utf-8-sig')
print("done")


# Calculate total carbon storage of point buffer and road based on point spatial join result field JOIN_ID
import pandas as pd

# Read the CSV file
df = pd.read_csv(results_folder +r"\线分割点C_exchange连接road_Split5000.csv",encoding='utf-8-sig')
# Group by group_field and calculate the sum of sum_field

# by_group_Carbon=df.groupby('JOIN_FID')['Carboni_1']
# by_group_Biomass=df.groupby('JOIN_FID')['Biomassi_1']

# sumBiomass_by_group = by_group_Biomass.sum().reset_index()
# sumCarbon_by_group = by_group_Carbon.sum().reset_index()


# Group by group_field, calculate sum of sum_field
# sumBiomass_by_group= df.groupby('JOIN_FID')['Biomassi+1'].sum().reset_index()
# sumCarbon_by_group = df.groupby('JOIN_FID')['Carboni+1'].sum().reset_index()
sumCarbon_exchange_by_group = df.groupby('JOIN_FID')['Carbon_exchange'].sum().reset_index()



# Save the results to new CSV files
# sumBiomass_by_group.to_csv(results_folder +r"\sumBiomassi+1_by_groupJOIN_FID.csv", index=False,encoding='utf-8-sig')
# sumCarbon_by_group.to_csv(results_folder +r"\sumCarboni+1_by_groupJOIN_FID.csv", index=False,encoding='utf-8-sig')

# Save the results to new CSV files
# sumBiomass_by_group.to_csv(results_folder +r"\sumBiomassi+1_by_groupNEAR_ID.csv", index=False,encoding='utf-8-sig')
# sumCarbon_by_group.to_csv(results_folder +r"\sumCarboni+1_by_groupNEAR_ID.csv", index=False,encoding='utf-8-sig')
sumCarbon_exchange_by_group.to_csv(results_folder +r"\road_Split5000sumC_exchange.csv", index=False,encoding='utf-8-sig')


## Calculate annual carbon sequestration based on ['Carboni+1'] and ['Carbon']
# import pandas as pd

# # Read the CSV files
# # df1 = pd.read_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final.csv")
# df2 = pd.read_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final_i+1.csv",encoding='utf-8-sig')

# # Add a new column to the original file
# df2['Carbon_exchange'] = df2['Carboni+1'] - df2['Carbon']

# # Save back to the original file
# df2.to_csv(r"E:\Suyingcai\changsha\答辩后修改\changsha_TreeMerge_final_i+1.csv", index=False,encoding='utf-8-sig')
