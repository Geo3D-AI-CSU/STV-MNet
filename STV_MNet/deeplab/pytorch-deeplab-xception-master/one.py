import os
import cv2
import numpy as np

original_folder = r"./"
new_folder = r'./res'

for filename in os.listdir(original_folder):
    path=os.path.join(original_folder, filename)
    img = cv2.imread(os.path.join(original_folder, filename), 1)
    print(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Binarization method: binarize the image using 127 as the threshold value
    retval, bit_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(new_folder, filename), bit_img)
    #112.902007783_28.1290705314_0_0_T.png