

import os
import glob
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
import tensorflow.contrib.slim as slim
import scipy.misc 
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

# 导入模型定义和参数
from monodepth_model import MonodepthModel, monodepth_parameters



#输入是一个视差图，输出后是经过后处理的视差图
def post_process_disparity(disp):
    #视差图的通道数，高度和宽度
    _, h, w = disp.shape
    #左视图的视差图
    l_disp = disp[0,:,:]
    #右视图的视差图
    r_disp = np.fliplr(disp[1,:,:])

    m_disp = 0.5 * (l_disp + r_disp)#取均值
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp#加权求和


# 设置模型参数
params = monodepth_parameters(
    encoder='vgg',
    height=256,
    width=512,
    batch_size=1,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode='border',
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False
)

# 定义图片路径
image_folder = r'Your street view images'

# 定义模型检查点路径
ROOT=r"E:\Suyingcai\STV_MNet"
checkpoint_path = ROOT+r"\code\Location\MonoDepth\data\model_cityscapes\model_cityscapes.meta"

# 创建模型对象
tf.reset_default_graph()
left = tf.placeholder(tf.float32, [2, params.height, params.width, 3])
model = MonodepthModel(params, 'test', left, None)

# 创建会话并加载模型
saver = tf.train.Saver()
sess = tf.Session()
restore_path = checkpoint_path.split(".")[0]
saver.restore(sess, restore_path)

# 遍历文件夹中的图像
for image_path in glob.glob(os.path.join(image_folder, '*.jpg')):
    # 读取图像
    input_image = scipy.misc.imread(image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [params.height, params.width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    #运行模型，得到视差图
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    #计算视差图
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    output_directory = ROOT+r"\results\Location calculation\monoDepth\changsha_Monodepth"
    output_name = os.path.splitext(os.path.basename(image_path))[0]
    
    #对视差图进行后处理，保存为numpy数组
    np.save(os.path.join(output_directory,'output_npy', "{}_disp.npy".format(output_name)), disp_pp)
    #根据数组转换为图像
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, 'output_png',"{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    #plasma用于色彩映射

# 关闭会话
sess.close()
print("done")
