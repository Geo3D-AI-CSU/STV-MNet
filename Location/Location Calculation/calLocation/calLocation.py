import math
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

ROOT = r"E:\Suyingcai\STV_MNet"
input_csv_folder = ROOT + r"\data\input data\Location\LNinput_csv"
input_npy_folder = ROOT + r"\results\Location calculation\monoDepth\changsha_Monodepth\output_npy"
output_csv_folder = ROOT + r'\results\Location calculation\output_csv'

# Calculate depth based on disparity
def caldepth(D):
    # Focal length F is 0.54 meters
    F = 0.54
    # W0 is set to 721 pixels
    W0 = 721
    # W1 is set to 6656 pixels
    W1 = 6656
    # Conversion factor C is set to 1.5
    C = 1.5
    return (W0 * F * C) / (W1 * D)

# Calculate angle using a given value
def calangle(a):
    # Half of the image width
    w = 64
    # sin(angle) value
    angle0 = math.sqrt(2) / 2
    angle1 = a * angle0 / w
    # Calculate the angle in radians
    angle1_radians = math.asin(angle1)
    print("sin(angle1)", math.sin(angle1_radians))
    # Convert radians to degrees
    angle1_degrees = math.degrees(angle1_radians)
    # 45 degrees angle
    anglesta = math.radians(45)
    result = anglesta - angle1_radians
    # Final result in degrees
    Result = math.degrees(result)
    return Result, math.sin(angle1_radians)

# Angle + 45°
def calangle1(a):
    # Half of the image width
    w = 64
    # sin(angle) value
    angle0 = math.sqrt(2) / 2
    angle1 = a * angle0 / w
    # Calculate the angle in radians
    angle1_radians = math.asin(angle1)
    # Convert radians to degrees
    angle1_degrees = math.degrees(angle1_radians)
    # 45 degrees angle
    anglesta = math.radians(45)
    result = anglesta + angle1_radians
    # Final result
    Result = math.degrees(result)
    return Result

# Record the initial time
tic = time.time()

# Main function for location calculation
def callocation(input_csv_folder, input_npy_folder, output_csv_folder):
    # Get all CSV filenames in the input folder
    csv_files = [file for file in os.listdir(input_csv_folder) if file.endswith('.csv')]

    for i, csv_file in enumerate(csv_files):
        if i % 10 == 0:
            print(i / 10, '    ', "Elapsed time:", '%.2f' % (time.time() - tic))
        
        # Construct the corresponding Numpy filename
        npy_file = os.path.splitext(csv_file)[0] + "_disp.npy"

        # Extract x and y coordinates from the filename
        coords = csv_file.split('_')[1].split(',')
        x = float(coords[0])
        y = float(coords[1])

        # Construct the full path for the input CSV file
        input_csv_path = os.path.join(input_csv_folder, csv_file)
        input_npy_path = os.path.join(input_npy_folder, npy_file)

        # Construct the full path for the output CSV file
        output_csv_path = os.path.join(output_csv_folder, csv_file)

        # Process the individual file
        process_single_file(input_csv_path, input_npy_path, output_csv_path, x, y)
        print("Finished processing the", i + 1, "th image")

# Function to process a single file
def process_single_file(input_csv_path, input_npy_path, output_csv_path, x, y):
    # Load CSV and Numpy data
    df = pd.read_csv(input_csv_path)
    data = np.load(input_npy_path)

    # Define arrays to store disparity values and offsets
    my_list = []
    x1 = []
    y1 = []

    # Define arrays to store the distance from the center line of the image
    dis = []

    # Define arrays for the result coordinates and angles
    resultx = []
    resulty = []
    resultdepth = []
    resultangles = []
    resultsangle1 = []

    for i in range(len(df)):
        # Read the necessary data
        data1 = float(df.iloc[i, 1])
        data2 = float(df.iloc[i, 2])
        data3 = float(df.iloc[i, 3]) / 4
        middle = (data1 + data2) / 2
        middle = middle / 4
        
        # The middle x value
        middlex = int(middle + 0.5)

        # y value
        data3 = int(data3 + 0.5)
        if data3 >= len(data):
            data3 = len(data) - 1
        
        # Append the disparity value
        my_list.append(data[data3][middlex])
        print("Disparity value for tree", i, ":")
        print(data[data3][middlex])

        # Calculate depth
        depth = caldepth(my_list[i])
        print("Absolute depth for tree", i, ":")
        print(depth)
        resultdepth.append(depth)

        # Determine the distance from the center line of the image
        if middle >= 0 and middle <= 64:
            distance = middle
        elif middle > 64 and middle <= 192:
            distance = abs(middle - 128)
        elif middle > 192 and middle <= 320:
            distance = abs(middle - 256)
        elif middle > 320 and middle <= 448:
            distance = abs(middle - 384)
        elif middle > 448 and middle <= 512:
            distance = abs(middle - 512)
        dis.append(distance)

        # Calculate angle
        resultangle, angle1 = calangle(distance)
        resultangles.append(resultangle)
        resultsangle1.append(angle1)

        # Calculate cos and sin values
        resultangle_radians = math.radians(resultangle)
        sin_value = math.sin(resultangle_radians)
        cos_value = math.cos(resultangle_radians)

        # Calculate the x and y offsets
        xD = depth * cos_value / 100000
        yD = depth * sin_value / 100000

        print("x offset for tree", i, ":")
        print(xD)
        print("y offset for tree", i, ":")
        print(yD)

        # Append the offsets to the arrays
        x1.append(xD)
        y1.append(yD)

        # Adjust the coordinates based on the middle position
        if (middle >= 256):
            resultx.append(x + xD)
            resulty.append(y + yD)
        elif (middle <= 128):
            resultx.append(x - xD)
            resulty.append(y - yD)
        else:
            resultx.append(x - xD)
            resulty.append(y + yD)

    # Save the results to the output CSV file
    output_csv_path = os.path.join(output_csv_folder, 'pro_' + os.path.basename(output_csv_path))
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'left', 'top', 'right', 'bottom', 'depth', 'sin(angle1)', 'x', 'y'])  # Write header
        for i in range(len(resultx)):
            writer.writerow([i, df.iloc[i, 1], df.iloc[i, 3], df.iloc[i, 2], df.iloc[i, 4], resultdepth[i], resultsangle1[i], resultx[i], resulty[i]])  # Write coordinate data

# Main function to trigger the location calculation
def main():
    callocation(input_csv_folder, input_npy_folder, output_csv_folder)

if __name__ == "__main__":
    main()
