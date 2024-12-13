import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

'''
To make your own semantic segmentation dataset you need to pay attention to the following points:
1, the version of labelme used is 3.16.7, it is recommended to use this version of labelme, some versions of labelme will occur error.
   Specific error: Too many dimensions: 3 > 2
   Installation for the command line pip install labelme==3.16.7
2、The labelme generated here is an 8-bit color map, which is not quite the same format as the dataset that looks like in the video.
   Although it looks like a color map, it is in fact only 8-bit, at this point the value of each pixel point is the kind that this pixel point belongs to.
   So it's actually the same format as the VOC dataset in the video. So the dataset made this way is working fine. It's also normal.

Translated with www.DeepL.com/Translator (free version)
'''
if __name__ == '__main__':
    image_path = "./datasets/JPEGImages"
    label_path = "./datasets/SegmentationClass"
    # classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    classes = ["_background_", "Tree"]

    count = os.listdir("./datasets/before/")
    # count = os.listdir(r"E:\Suyingcai\StreetView\ultralytics-main\ultralytics-main\runs\segment\predict4\labelme_json/")
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])
        # path = os.path.join(r"E:\Suyingcai\StreetView\ultralytics-main\ultralytics-main\runs\segment\predict4\labelme_json/", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            print(path)
            data = json.load(open(path,encoding='UTF-8'))

            if data['imageData']:
                imageData = data['imageData']
            else:
                # print(os.path.dirname(path))
                # print(os.path.basename(path)[:-5]+".jpg")
                imagePath = os.path.join(os.path.dirname(path), os.path.basename(path)[:-5]+".jpg")
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(image_path, count[i][:-5] + '.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)
            utils.lblsave(osp.join(label_path, count[i][:-5] ), new)
            print('Saved ' + count[i] + '.jpg and ' + count[i] + '.jpg')
