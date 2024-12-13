# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
# OS-related functionality
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
# For parsing command-line arguments
import argparse
import re
import time

import tensorflow as tf 
import tensorflow.contrib.slim as slim
import scipy.misc 
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

# Create an argument parser object
parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='Type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='Path to the image',
                   default=r"E:\dong\file\data\464039_112.931781683,28.168779497_201908_71.286.jpg")

parser.add_argument('--checkpoint_path',  type=str,   help='Path to a specific checkpoint to load',
                    default=r"E:\dong\file\data\model_cityscapes\model_cityscapes.meta")
parser.add_argument('--input_height',     type=int,   help='Input height', default=256)
parser.add_argument('--input_width',      type=int,   help='Input width', default=512)

# Parse command-line arguments and store results in the variable
args = parser.parse_args()

# Helper function (processing disparity map)
# Input is a disparity map, output is the post-processed disparity map
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

# Test function
def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    # Create an object, load the model, and run the test
    model = MonodepthModel(params, "test", left, None)
    # Load input object and preprocess
    input_image = scipy.misc.imread(args.image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    train_saver = tf.train.Saver()

    # Create a session object to run the model
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # SAVER
    # Load model parameters


    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # Save
    restore_path = args.checkpoint_path.split(".")[0]
    # restore_path = r'E:\dong\file\data\model_cityscapes\model_cityscapes.meta'
    train_saver.restore(sess, restore_path)
    # Run the model to get the disparity map
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    # Process the disparity map
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    output_directory = r'E:\dong\file\data\model_cityscapes'
    output_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Post-process the disparity map and save it as a numpy array
    np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    # Convert the array to an image
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    # "plasma" is used for colormap

    print('done!')

def main(_):
    # Create an object and set model parameters
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    test_simple(params)

if __name__ == '__main__':
   tf.app.run()
