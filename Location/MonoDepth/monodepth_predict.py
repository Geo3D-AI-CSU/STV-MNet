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

# Import model definition and parameters
from monodepth_model import MonodepthModel, monodepth_parameters


# The input is a disparity map, and the output is the post-processed disparity map
def post_process_disparity(disp):
    # Number of channels, height, and width of the disparity map
    _, h, w = disp.shape
    # Left view disparity map
    l_disp = disp[0,:,:]
    # Right view disparity map
    r_disp = np.fliplr(disp[1,:,:])

    m_disp = 0.5 * (l_disp + r_disp)  # Take the average
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp  # Weighted sum


# Set model parameters
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

# Define the image folder path
image_folder = r'Your street view images'

# Define the model checkpoint path
ROOT = r"E:\Suyingcai\STV_MNet"
checkpoint_path = ROOT + r"\code\Location\MonoDepth\data\model_cityscapes\model_cityscapes.meta"

# Create model object
tf.reset_default_graph()
left = tf.placeholder(tf.float32, [2, params.height, params.width, 3])
model = MonodepthModel(params, 'test', left, None)

# Create a session and load the model
saver = tf.train.Saver()
sess = tf.Session()
restore_path = checkpoint_path.split(".")[0]
saver.restore(sess, restore_path)

# Iterate through the images in the folder
for image_path in glob.glob(os.path.join(image_folder, '*.jpg')):
    # Read the image
    input_image = scipy.misc.imread(image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [params.height, params.width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # Run the model to obtain the disparity map
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    # Process the disparity map
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    output_directory = ROOT + r"\results\Location calculation\monoDepth\changsha_Monodepth"
    output_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Post-process the disparity map and save it as a numpy array
    np.save(os.path.join(output_directory, 'output_npy', "{}_disp.npy".format(output_name)), disp_pp)
    # Convert the array to an image
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, 'output_png', "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    # "plasma" is used for colormap

# Close the session
sess.close()
print("done")
