"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.

Majority of this file is from the sample defense provided at:
https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf

import inception_resnet_v2

from scipy.ndimage.filters import median_filter
from fastaniso import anisodiff
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.restoration import denoise_nl_means

from skimage.util import random_noise

from PIL import Image
import io

from scipy.stats import mode

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def jpeg(image):
  buffer = io.BytesIO()
  im1 = Image.fromarray(image)
  im1.save(buffer, "JPEG", quality=9)
  jpegImg = imread(buffer, mode='RGB').astype(np.float) / 255.0

  return jpegImg

def nlm(image):
  nlnImg = denoise_nl_means(image, 7, 9, 0.08, multichannel=True)
  return nlnImg

def median(image):
  medianImg = np.zeros_like(image,dtype=np.float64)
  for i in range(3):
    medianImg[:,:,i] = median_filter(image[:,:,i], size=(3,3))

  return medianImg

def wavelet(image):
  im_bayes = denoise_wavelet(image, multichannel=True, convert2ycbcr=True, mode='soft')
  return im_bayes

def loadAllModes(input_dir, batch_shape, method=0):
  images1 = np.zeros(batch_shape)
  images2 = np.zeros(batch_shape)
  images3 = np.zeros(batch_shape)
  images4 = np.zeros(batch_shape)
  images5 = np.zeros(batch_shape)
  images6 = np.zeros(batch_shape)

  images1b = np.zeros(batch_shape)
  images2b = np.zeros(batch_shape)
  images3b = np.zeros(batch_shape)
  images4b = np.zeros(batch_shape)
  images5b = np.zeros(batch_shape)
  images6b = np.zeros(batch_shape)

  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image1 = imread(f, mode='RGB')
      image2 = image1.copy()
      image3 = image1.copy()
      image4 = image1.copy()
      image5 = image1.copy()
      image6 = image1.copy()

      image1b = image1.copy()

      sigma = 0.12
      noisy_images = random_noise(image1b, var=sigma**2)

      image1b = np.clip(noisy_images, 0.0, 1.0)

      image2b = image1b.copy()
      image3b = image1b.copy()
      image4b = image1b.copy()
      image5b = image1b.copy()
      image6b = image1b.copy()

      image1[:,:,0] = anisodiff(image1[:,:,0], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
      image1[:,:,1] = anisodiff(image1[:,:,1], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
      image1[:,:,2] = anisodiff(image1[:,:,2], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
    
      image2 = median(image2)

      image3 = jpeg(image3)

      image4 = wavelet(image4)

      image6 = nlm(image6)


      image1b[:,:,0] = anisodiff(image1b[:,:,0], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
      image1b[:,:,1] = anisodiff(image1b[:,:,1], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
      image1b[:,:,2] = anisodiff(image1b[:,:,2], niter=4,kappa=941.000000,gamma=0.250000,step=(2.000000,2.000000),option=1)
    
      image2b = median(image2b)
      image3b = jpeg(image3b.astype('uint8'))
      image4b = wavelet(image4b)
      image6b = nlm(image6b)


      image1 = image1.astype(np.float) / 255.0
      image2= image2.astype(np.float) / 255.0
      image3 = image3.astype(np.float)
      image4 = image4.astype(np.float)
      image5 = image5.astype(np.float) / 255.0
      image6 = image6.astype(np.float) / 255.0

      image1b = image1b.astype(np.float) / 255.0
      image2b= image2b.astype(np.float) / 255.0
      image3b = image3b.astype(np.float)
      image4b = image4b.astype(np.float)
      image5b = image5b.astype(np.float) / 255.0
      image6b = image6b.astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images1[idx, :, :, :] = image1 * 2.0 - 1.0
    images2[idx, :, :, :] = image2 * 2.0 - 1.0
    images3[idx, :, :, :] = image3 * 2.0 - 1.0
    images4[idx, :, :, :] = image4 * 2.0 - 1.0
    images5[idx, :, :, :] = image5 * 2.0 - 1.0
    images6[idx, :, :, :] = image6 * 2.0 - 1.0

    images1b[idx, :, :, :] = image1b * 2.0 - 1.0
    images2b[idx, :, :, :] = image2b * 2.0 - 1.0
    images3b[idx, :, :, :] = image3b * 2.0 - 1.0
    images4b[idx, :, :, :] = image4b * 2.0 - 1.0
    images5b[idx, :, :, :] = image5b * 2.0 - 1.0
    images6b[idx, :, :, :] = image6b * 2.0 - 1.0

    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images1, images2, images3, images4, images5, images6, images1b, images2b, images3b, images4b, images5b, images6b
      filenames = []
      images1 = np.zeros(batch_shape)
      images2 = np.zeros(batch_shape)
      images3 = np.zeros(batch_shape)
      images4 = np.zeros(batch_shape)
      images5 = np.zeros(batch_shape)
      images6 = np.zeros(batch_shape)

      images1b = np.zeros(batch_shape)
      images2b = np.zeros(batch_shape)
      images3b = np.zeros(batch_shape)
      images4b = np.zeros(batch_shape)
      images5b = np.zeros(batch_shape)
      images6b = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images1, images2, images3, images4, images5, images6, images1b, images2b, images3b, images4b, images5b, images6b

def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
  #with tf.variable_scope("InceptionResnetV2") as scope:
    # Prepare graph
    x_input1 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input2 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input3 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input4 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input5 = tf.placeholder(tf.float32, shape=batch_shape)
    x_input6 = tf.placeholder(tf.float32, shape=batch_shape)

    x_input1b = tf.placeholder(tf.float32, shape=batch_shape)
    x_input2b = tf.placeholder(tf.float32, shape=batch_shape)
    x_input3b = tf.placeholder(tf.float32, shape=batch_shape)
    x_input4b = tf.placeholder(tf.float32, shape=batch_shape)
    x_input5b = tf.placeholder(tf.float32, shape=batch_shape)
    x_input6b = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()) as scope:
      _, end_points1 = inception_resnet_v2.inception_resnet_v2(
          x_input1, num_classes=num_classes, is_training=False)

      _, end_points2 = inception_resnet_v2.inception_resnet_v2(
          x_input2, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points3 = inception_resnet_v2.inception_resnet_v2(
          x_input3, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points4 = inception_resnet_v2.inception_resnet_v2(
          x_input4, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points5 = inception_resnet_v2.inception_resnet_v2(
          x_input5, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points6 = inception_resnet_v2.inception_resnet_v2(
          x_input6, num_classes=num_classes, is_training=False, reuse=True)


      _, end_points1b = inception_resnet_v2.inception_resnet_v2(
          x_input1b, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points2b = inception_resnet_v2.inception_resnet_v2(
          x_input2b, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points3b = inception_resnet_v2.inception_resnet_v2(
          x_input3b, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points4b = inception_resnet_v2.inception_resnet_v2(
          x_input4b, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points5b = inception_resnet_v2.inception_resnet_v2(
          x_input5b, num_classes=num_classes, is_training=False, reuse=True)

      _, end_points6b = inception_resnet_v2.inception_resnet_v2(
          x_input6b, num_classes=num_classes, is_training=False, reuse=True)

    predicted_labels1 = tf.argmax(end_points1['Predictions'], 1)
    predicted_labels2 = tf.argmax(end_points2['Predictions'], 1)
    predicted_labels3 = tf.argmax(end_points3['Predictions'], 1)
    predicted_labels4 = tf.argmax(end_points4['Predictions'], 1)
    predicted_labels5 = tf.argmax(end_points5['Predictions'], 1)
    predicted_labels6 = tf.argmax(end_points6['Predictions'], 1)

    predicted_labels1b = tf.argmax(end_points1b['Predictions'], 1)
    predicted_labels2b = tf.argmax(end_points2b['Predictions'], 1)
    predicted_labels3b = tf.argmax(end_points3b['Predictions'], 1)
    predicted_labels4b = tf.argmax(end_points4b['Predictions'], 1)
    predicted_labels5b = tf.argmax(end_points5b['Predictions'], 1)
    predicted_labels6b = tf.argmax(end_points6b['Predictions'], 1)


    predicted_labels = tf.stack([predicted_labels1,predicted_labels2,predicted_labels3,predicted_labels4,predicted_labels6,predicted_labels1,predicted_labels2,predicted_labels3,predicted_labels4,predicted_labels6,predicted_labels5,
      predicted_labels1b,predicted_labels2b,predicted_labels3b,predicted_labels4b,predicted_labels5b,predicted_labels6b])

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        #labelList = [[],[],[]]
        #fileList = []

        for filenames, images1, images2, images3, images4, images5, images6, images1b, images2b, images3b, images4b, images5b, images6b in loadAllModes(FLAGS.input_dir, batch_shape):

          labels = sess.run(predicted_labels, feed_dict={x_input1: images1, x_input2: images2, x_input3: images3, x_input4: images4, x_input5: images5, x_input6: images6, x_input1b: images1b, x_input2b: images2b, x_input3b: images3b, x_input4b: images4b, x_input5b: images5b, x_input6b: images6b})

          labels = mode( labels, axis=0).mode.tolist()

          for filename, label in zip(filenames, labels[0]):#labels[0]):
            out_file.write('{0},{1}\n'.format(filename, label))

          


if __name__ == '__main__':
  tf.app.run()
