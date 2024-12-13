from ultralytics.models import YOLO,RTDETR
import numpy as np
from PIL import Image
import glob
import cv2
import os


def tojson(r, normalize=False):
    """Convert the object to JSON format."""

    import json

    # Create list of detection dictionaries
    results = []
    data = r.boxes.data.cpu().tolist()
    h, w = r.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
        box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
        conf = row[-2]
        class_id = int(row[-1])
        name = r.names[class_id]
        result = {'name': name, 'class': class_id, 'confidence': conf, 'box': box}
        if r.boxes.is_track:
            result['track_id'] = int(row[-3])  # track ID
        if r.masks:
            x, y = r.masks.xy[i][:, 0], r.masks.xy[i][:, 1]  # numpy array
            result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
        if r.keypoints is not None:
            x, y, visible = r.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
            result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
        results.append(result)

    # Convert detections to JSON
    return json.dumps(results, indent=2,separators=(',', ':'))


if __name__ == '__main__':
    STV_MNetPath=r'E:\Suyingcai\STV_MNet'
    model = YOLO(r'runs\segment\split_3_datasetODConv_3rd_WIoU\weights\best.pt')
    
    results = model.predict(
    source="Your original street view images", 
    # The source directory or file for the images or videos to be processed
    
    stream=True,  
    # Use stream mode for video processing to prevent memory overflow due to frame accumulation
    
    # show=True,  
    # Enable real-time inference demonstration
    
    data="data/data.yaml",  
    # Configuration file specifying model parameters and dataset information
    
    save=True, 
    # Choose whether to save the results
    
    # vid_stride=2,  
    # Frame stride for video, i.e., how many frames to skip between detections/tracking
    
    save_txt=True,  
    # Save results in text format
    
    # save_conf=True,  
    # Save confidence scores
    
    # save_crop=True,  
    # Save cropped images of detected objects
    
    conf=0.1, 
    # Confidence threshold; detections below this threshold will be discarded
    
    iou=0.7,  
    # IoU (Intersection over Union) threshold for NMS (Non-Maximum Suppression) to remove redundant bounding boxes for the same object
    
    device="0,1", 
    # Use GPU for inference, specify which GPUs to use. Use "cpu" for CPU processing
)

# Process the results for each frame returned by the model

for r in results:
    filepath=r.save_dir+"/json_seg"
    if os.path.exists(filepath) ==False:
            os.makedirs(filepath)
    filename=os.path.basename(r.path)
    json_file=filepath+"/{}.json".format(filename[:-4])
    json_data=tojson(r)
    # print(json_data)
    with open(json_file, 'w') as file:
        file.write(json_data)

        

