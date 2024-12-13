import os
import os.path as osp
import shutil
import cv2
import json
import numpy as np

# Dictionary mapping class names to class indices
label_dict = {
    "Tree": 0
}

# Dictionary mapping class indices to colors for visualization
color_dict = {
    0: (255, 0, 0)  # Red for trees
}

# Define root directory
ROOT = r"E:\Suyingcai\STV_MNet\code"
BASEDIR = osp.join(ROOT, r"STV_MNet\yolov3-master\data\labelsSegment")
# BASEDIR = r"E:\Suyingcai\StreetView\ultralytics-main\ultralytics-main\runs\segment\predict"
IMGDIR = osp.join(BASEDIR, 'png')  # Image directory
LABDIR = osp.join(BASEDIR, 'bbox')  # Label directory
OUTDIR = osp.join(BASEDIR, 'json_img')  # Output directory
print("IMGDIR**********:", IMGDIR)

# Remove the output directory if it exists, then create it
if osp.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)

# List all image files with specific extensions
imgnames = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']]  # Filter by extension
print("Image names:", imgnames)

for imgname in imgnames:
    path_img = osp.join(IMGDIR, imgname)
    img = cv2.imread(path_img)  # Read the image
    file_name, file_ext = os.path.splitext(imgname)
    print("JSON names:", file_name)
    path_json = osp.join(LABDIR, file_name + '.json')  # Find the corresponding JSON annotation file
    
    if os.path.exists(path_json):
        with open(path_json, 'r') as fp:
            jsonData = json.load(fp)
        
        boxes = jsonData["shapes"]  # Get the list of shapes from JSON
        for box in boxes:
            cls_name = box["label"]  # Class name
            xy4 = box["points"]  # Points of the bounding box
            xy4 = np.array(xy4, dtype=np.int0)  # Convert to integer array (precision loss!)
            # print(xy4.shape, xy4.dtype)  # shape: (4, 2) /*four 2D coordinates*/, dtype: int64 /*integer type*/
            
            # Draw contours on the image
            cv2.drawContours(img, [xy4], 0, color_dict[label_dict[cls_name]][::-1], 2)  # Draw the bounding box
            
        cv2.imwrite(osp.join(OUTDIR, imgname), img)  # Save the image with drawn contours
        print(imgname + ", done...")
