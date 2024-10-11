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
        source="Your original street view images", # 
        stream=True,  # 对于视频采用流模式处理，防止因为因为堆积而内存溢出
        # show=True,  # 实时推理演示
        data="data/data.yaml",  # 
        save=True, # 选择是否保存
        # vid_stride=2,  # 视频帧数的步长，即隔几帧检测跟踪一次
        save_txt=True,  # 把结果以txt形式保存
        # save_conf=True,  # 保存置信度得分
        # save_crop=True,  # 保存剪裁的图像
        conf=0.1, # 规定阈值，即低于该阈值的检测框会被剔除
        iou=0.7, # 交并比阈值，用于去除同一目标的冗余框
        device="0,1", # 用GPU进行推理，如果使用cpu，则为device="cpu"
    )
    
    # 对每一帧返回的结果进行处理
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

        

