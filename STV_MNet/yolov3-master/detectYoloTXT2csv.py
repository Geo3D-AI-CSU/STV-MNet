import os
import csv

# Define a function to convert center point coordinates and width/height to top-left and bottom-right coordinates
def xywh_to_left_top_right_bottom(xywh, image_width=2048, image_height=1024):
    center_x, center_y, width, height = xywh
    left = (center_x - width / 2) * image_width
    top = (center_y - height / 2) * image_height
    right = (center_x + width / 2) * image_width
    bottom = (center_y + height / 2) * image_height
    return left, top, right, bottom

# Input folder path
input_folder = r'E:\Suyingcai\StreetView\ultralytics_miou\ultralytics_miou\runs\segment\predict_lunan\box_labels'
# Output folder path
output_folder = r'E:\Suyingcai\StreetView\ultralytics_miou\ultralytics_miou\runs\segment\predict_lunan\csv'

# Iterate through each txt file and generate a corresponding csv file
for txt_file_name in os.listdir(input_folder):
    # Check if the file extension is txt
    if txt_file_name.endswith('.txt'):
        txt_file_path = os.path.join(input_folder, txt_file_name)
        csv_file_path = os.path.join(output_folder, os.path.splitext(txt_file_name)[0] + '.csv')

        # Create and write to CSV file
        with open(txt_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
            txt_reader = csv.reader(txt_file, delimiter=' ')
            csv_writer = csv.writer(csv_file)

            # Write CSV header
            csv_writer.writerow(["ID", "left", "top", "right", "bottom", "class"])

            # Parse each line and write data to CSV file
            for idx, line in enumerate(txt_reader):
                # cls, center_x, center_y, width, height,score = map(float, line)
                cls, center_x, center_y, width, height = map(float, line)
                left, top, right, bottom = xywh_to_left_top_right_bottom((center_x, center_y, width, height))
                csv_writer.writerow([idx, left, top, right, bottom, "Tree"])
