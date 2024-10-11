import os
import cv2
import numpy as np

# 原始文件夹路径
original_folder = r"./"
# 保存的新文件夹路径
new_folder = r'./res'
# 遍历原始文件夹中的图像
for filename in os.listdir(original_folder):
    path=os.path.join(original_folder, filename)
    img = cv2.imread(os.path.join(original_folder, filename), 1)
    print(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #二值化方法：以127做为阈值进行二值化图像
    retval, bit_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(new_folder, filename), bit_img)
    #112.902007783_28.1290705314_0_0_T.png