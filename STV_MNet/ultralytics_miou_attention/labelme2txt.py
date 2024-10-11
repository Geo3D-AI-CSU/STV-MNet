import base64
import random
import shutil
from tqdm import tqdm
import math
import json
import os
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2
 
 
class ConvertManager(object):
    def __init__(self):
        pass
 
    def base64_to_numpy(self, img_bs64):
        img_bs64 = base64.b64decode(img_bs64)
        img_array = np.frombuffer(img_bs64, np.uint8)
        cv2_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2_img
 
    @classmethod
    def load_labels(cls, name_file):
        '''
        load names from file.one name one line
        :param name_file:
        :return:
        '''
        with open(name_file, 'r') as f:
            lines = f.read().rstrip('\n').split('\n')
        return lines
 
    def get_class_names_from_all_json(self, json_dir):
        classnames = []
        for file in os.listdir(json_dir):
            if not file.endswith('.json'):
                continue
            with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                for shape in data_dict['shapes']:
                    if not shape['label'] in classnames:
                        classnames.append(shape['label'])
        return classnames
 
    def create_save_dir(self, save_dir):
        images_dir = os.path.join(save_dir, 'images')
        labels_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.mkdir(images_dir)
            os.mkdir(labels_dir)
        else:
            if not os.path.exists(images_dir):
                os.mkdir(images_dir)
            if not os.path.exists(labels_dir):
                os.mkdir(labels_dir)
        return images_dir + os.sep, labels_dir + os.sep
 
    def save_list(self, data_list, save_file):
        with open(save_file, 'w') as f:
            f.write('\n'.join(data_list))
 
    def __rectangle_points_to_polygon(self, points):
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        if points[0][0] > points[1][0]:
            xmax = points[0][0]
            ymax = points[0][1]
            xmin = points[1][0]
            ymin = points[1][1]
        else:
            xmax = points[1][0]
            ymax = points[1][1]
            xmin = points[0][0]
            ymin = points[0][1]
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
 
    def convert_dataset(self, json_dir, json_list, images_dir, labels_dir, names, save_mode='train'):
        images_dir = os.path.join(images_dir, save_mode)+os.sep
        labels_dir = os.path.join(labels_dir, save_mode)+os.sep
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
        for file in tqdm(json_list):
            filePath=os.path.join(json_dir, file)
            with open(filePath, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            image_file = os.path.join(json_dir, os.path.basename(filePath[:-5]+".jpg"))
            if os.path.exists(image_file):
                shutil.copyfile(image_file, images_dir + os.path.basename(image_file))
            else:
                imageData = data_dict.get('imageData')
                if not imageData:
                    imageData = base64.b64encode(imageData).decode('utf-8')
                    img = self.img_b64_to_arr(imageData)
                    PIL.Image.fromarray(img).save(images_dir + file[:-4] + 'png')
            # convert to txt
            width = data_dict['imageWidth']
            height = data_dict['imageHeight']
            line_list = []
            for shape in data_dict['shapes']:
                data_list = []
                data_list.append(str(names.index(shape['label'])))
                if shape['shape_type'] == 'rectangle':
                    points = self.__rectangle_points_to_polygon(shape['points'])
                    for point in points:
                        data_list.append(str(point[0] / width))
                        data_list.append(str(point[1] / height))
 
 
                elif shape['shape_type'] == 'polygon':
                    points = shape['points']
                    for point in points:
                        data_list.append(str(point[0] / width))
                        data_list.append(str(point[1] / height))
                line_list.append(' '.join(data_list))
 
            self.save_list(line_list, labels_dir + file[:-4] + "txt")
 
    def split_train_val_test_dataset(self, file_list, train_ratio=0.9, trainval_ratio=0.9, need_test_dataset=False,
                                     shuffle_list=True):
        if shuffle_list:
            random.shuffle(file_list)
        total_file_count = len(file_list)
        train_list = []
        val_list = []
        test_list = []
        if need_test_dataset:
            trainval_count = int(total_file_count * trainval_ratio)
            trainval_list = file_list[:trainval_count]
            test_list = file_list[trainval_count:]
            train_count = int(train_ratio * len(trainval_list))
            train_list = trainval_list[:train_count]
            val_list = trainval_list[train_count:]
        else:
            train_count = int(train_ratio * total_file_count)
            train_list = file_list[:train_count]
            val_list = file_list[train_count:]
        return train_list, val_list, test_list
 
    def start(self, json_dir, save_dir, names=None, train_ratio=0.7):
        images_dir, labels_dir = self.create_save_dir(save_dir)
        if names is None or len(names) == 0:
            print('class names will load from all json file')
            names = self.get_class_names_from_all_json(json_dir)
        print('find {} class names :'.format(len(names)), names)
        if len(names) == 0:
            return
 
        self.save_list(names, os.path.join(save_dir, 'labels.txt'))
        print('start convert')
        all_json_list = []
        for file in os.listdir(json_dir):
            if not file.endswith('.json'):
                continue
            all_json_list.append(file)
        # train_list, val_list, test_list = self.split_train_val_test_dataset(all_json_list, train_ratio)
        # self.convert_dataset(json_dir, train_list, images_dir, labels_dir, names, 'train')
        # self.convert_dataset(json_dir, val_list, images_dir, labels_dir, names, 'val')
        self.convert_dataset(json_dir, all_json_list, images_dir, labels_dir, names, 'trainval')
 
 
if __name__ == '__main__':
    STV_MNetPath=r'E:\Suyingcai\STV_MNet'
    STV_MNetData=STV_MNetPath+r"\data\input data\STV_MNet"
    cm = ConvertManager()
    cm.start(STV_MNetData+r'\data\trainval\json', STV_MNetData+r'\data\trainval\txt')
    # cm.convert_dataset(json_dir, val_list, images_dir, labels_dir, names, 'val')