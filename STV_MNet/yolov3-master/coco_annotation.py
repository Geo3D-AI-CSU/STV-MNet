

# *_*coding: utf-8 *_*
# Author --LiMing--

import os
import random
import shutil
import time


def copyFile(fileDir, origion_path1, class_name):
    name = class_name#按顺序读取到第一张照片名字
    path = origion_path1#标签地址
    image_list = os.listdir(fileDir)  # 获取原始图片路径中的所有图片
    image_number = len(image_list)
    train_number = int(image_number * train_rate)#图像数量乘以随机比例得出需要多少张训练图像
    test_number = int(image_number * test_rate)#图像数量乘以随机比例得出需要多少张验证集图像
    train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取0.8比例的图像.
    test_sample = random.sample(list(set(image_list) - set(train_sample)), test_number)
    val_sample = list(set(image_list ) - set(train_sample) - set(test_sample))
    #test_sample = list(set(image_list) - set(train_sample))#测试集中保留剩余图像

    sample = [train_sample, test_sample, val_sample]#生成三个列表
    # 复制图像到目标文件夹
    for k in range(len(save_dir)):#地址长度，目前k是两个数，0和1
        if os.path.isdir(save_dir[k]) and os.path.isdir(save_dir1[k]):#判断路径是否存在
            for name in sample[k]:#sample[0]为train_sample中的数据，整句是train_sample中的数据循序进行遍历
                name1 = name[0:-4] + '.txt'#split()的用处是拆分字符串，1927.jpg  以 .  开始拆分成‘1927’，‘jpg’  [0]是指只用1927这个字符后面加.txt
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))#join的作用是拼接字符串，
                shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))#copy的作用是复制
        else:
            os.makedirs(save_dir[k])#建立路径图像
            os.makedirs(save_dir1[k])#建立路径标签
            for name in sample[k]:
                name1 = name[0:-4] + '.txt'
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
                shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))
        print('ok')


if __name__ == '__main__':
    time_start = time.time()

    # 原始数据集路径
    origion_path = r'E:\syc\StreetView\yolov3-master\labels\png/'#img
    origion_path1 = r'E:\syc\StreetView\yolov3-master\labels\txt/'#labels

    # 保存路径
    save_train_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\train\images/'#训练集图片地址
    save_test_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\test\images/'#测试集图片地址
    save_val_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\val\images/'  # 验证集图片地址
    save_train_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\train\labels/'#训练集标签
    save_test_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\test\labels/'#测试集标签
    save_val_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\val\labels/'  # 验证集标签
    save_dir = [save_train_dir, save_test_dir, save_val_dir]
    save_dir1 = [save_train_dir1, save_test_dir1, save_val_dir1]

    # 训练集比例
    train_rate = 0.8
    test_rate = 0.2

    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)
    for i in range(num_classes):
        class_name = file_list[i]
        copyFile(origion_path, origion_path1, class_name)
    print('划分完毕!')
    time_end = time.time()
    print('---------------')
    print('训练集和测试集划分共耗时%s!' % (time_end - time_start))

