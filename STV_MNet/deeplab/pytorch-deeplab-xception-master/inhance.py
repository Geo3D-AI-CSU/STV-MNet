#!/usr/bin/env python
# -*- coding: utf-8 -*-

# !/usr/bin/env python

import sys
import cv2
import os
from PIL import Image
from PIL import ImageDraw

os.getcwd()
#path = os.path.join(os.getcwd(), 'image')
path = r"E:\tao\tao\deeplearning\SegmentationClass"
print(path)
im_file = os.listdir(path)

for im_file_index_i in im_file:
    #os.getcwd()
    #path = os.path.join(os.getcwd(), 'image')
    # im_test = os.path.join(path, im_file_index_i)
    im_test = os.path.join(path, im_file_index_i)
    print(im_test)
    img1 = Image.open(im_test)

    # ====================================================
    out = Image.open(im_test)
    out1 = out.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转

    # out3 = out.rotate(45)                            #45°顺时针翻转
    # out4 = img.rotate(30)                            #30°顺时针翻转
    # out1.show()
    # out2.show()
    # out3.show()
    # out4.show()
    name1 = path+ '\T' + im_file_index_i
    out1.save(name1)

