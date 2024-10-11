import os
import os.path as osp
import shutil
import cv2
import json
import numpy as np

label_dict = {
    "Tree":0
}

color_dict = {
    0: (255, 000, 000)
}
ROOT=r"E:\Suyingcai\STV_MNet\code"
BASEDIR = ROOT+r"\STV_MNet\yolov3-master\data\labelsSegment"
# BASEDIR = r"E:\Suyingcai\StreetView\ultralytics-main\ultralytics-main\runs\segment\predict"
IMGDIR = osp.join(BASEDIR, 'png')
LABDIR = osp.join(BASEDIR, 'bbox')
OUTDIR = osp.join(BASEDIR, 'json_img')
print("IMGDIR**********:",IMGDIR)
if osp.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)

imgnames = [name for name in os.listdir(IMGDIR) if name.split('.')[-1] in ['png', 'jpg', 'tif', 'bmp']]  # 按扩展名过滤
print("imagenames:",imgnames)
for imgname in imgnames:
    path_img = osp.join(IMGDIR, imgname)
    img = cv2.imread(path_img)  # 读图
    file_name, file_ext = os.path.splitext(imgname)
    print("jsonnames:",file_name)
    path_json = osp.join(LABDIR, file_name + '.json')  # 找到对应名字的json标注文件
    if os.path.exists(path_json):
        with open(path_json, 'r') as fp:
            jsonData = json.load(fp)
        boxes = jsonData["shapes"]
        for box in boxes:
            cls_name = box["label"]
            xy4 = box["points"]
            xy4 = np.array(xy4, dtype=np.int0)  # 损失精度！
            # print(xy4.shape, xy4.dtype)  # shape: (4, 2) /*四个二维坐标*/, dtype: int64 /*整型*/
            # 图纸，点阵集，索引，颜色，粗细
            cv2.drawContours(img, [xy4], 0, color_dict[label_dict[cls_name]][::-1], 2)  # 画边框
        cv2.imwrite(osp.join(OUTDIR, imgname), img)
        print(imgname + ", done...")
