import os
import random
import shutil

trainval_percent = 0.3
train_percent = 0.7
ROOT=r"E:\Suyingcai\STV_MNet\code"
yoloBASEDIR = ROOT+r"\STV_MNet\yolov3-master\data\labelsSegment"
yolo_txt_dir = yoloBASEDIR+r'\MyCOCO'
txtfilepath = yoloBASEDIR+r'\txt'
total_txt = os.listdir(txtfilepath)

num = len(total_txt)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(yolo_txt_dir+r'\trainval.txt', 'w')
ftest = open(yolo_txt_dir+r'\test.txt', 'w')
ftrain = open(yolo_txt_dir+r'\train.txt', 'w')
fval = open(yolo_txt_dir+r'\val.txt', 'w')

for i in list:
    name = total_txt[i]
    original_txt = os.path.join(txtfilepath, total_txt[i])
    print(original_txt)
    original_img = os.path.join(yoloBASEDIR+r'\png', total_txt[i][:-4] + '.jpg')
    if i in trainval:
        new_imgPath = yolo_txt_dir +r'\images\trainval'
        new_labelPath = yolo_txt_dir +r'\labels\trainval'
        new_txt = os.path.join(new_labelPath, name)
        new_img = os.path.join(new_imgPath, total_txt[i][:-4] + '.png')
        ftrainval.write(new_txt+ '\n')
        shutil.copy(original_txt, new_txt)
        shutil.copy(original_img, new_img)
        if i in train:
            new_imgPath = yolo_txt_dir +r'\images\test'
            new_labelPath = yolo_txt_dir +r'\labels\test'
            new_txt = os.path.join(new_labelPath, name)
            new_img = os.path.join(new_imgPath, total_txt[i][:-4] + '.png')
            ftest.write(new_txt+ '\n')
            shutil.copy(original_txt, new_txt)
            shutil.copy(original_img, new_img)
        else:
            new_imgPath = yolo_txt_dir +r'\images\val'
            new_labelPath = yolo_txt_dir +r'\labels\val'
            new_txt = os.path.join(new_labelPath, name)
            new_img = os.path.join(new_imgPath, total_txt[i][:-4] + '.png')
            fval.write(new_txt+ '\n')
            shutil.copy(original_txt, new_txt)
            shutil.copy(original_img, new_img)
    else:
        new_imgPath = yolo_txt_dir + r'\images\train'
        new_labelPath =yolo_txt_dir + r'\labels\train'
        new_txt = os.path.join(new_labelPath, name)
        new_img = os.path.join(new_imgPath, total_txt[i][:-4] + '.png')
        ftrain.write(new_txt+ '\n')
        shutil.copy(original_txt, new_txt)
        shutil.copy(original_img, new_img)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

