from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat


IMAGE_SIZE = 320
LABEL_SIZE = 80

NUM_CLASSES = 2060  # 3307
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 300

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

def read_data(queue):
    class DATARecord(object):
        pass

    result = DATARecord()

    image = tf.read_file(queue[0])
    result.uint8image = tf.image.decode_jpeg(image, channels=3)
    shape = tf.shape(result.uint8image)

    label = tf.read_file(queue[1])
    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, [shape[1], shape[0]])
    label = tf.transpose(label)
    result.label = label

    edge = tf.read_file(queue[2])
    edge = tf.decode_raw(edge, tf.uint8)
    edge = tf.reshape(edge, [shape[1], shape[0]])
    edge = tf.transpose(edge)
    result.edge = edge

    poss_lbls = tf.read_file(queue[3])
    poss_lbls = tf.decode_raw(poss_lbls, tf.uint8)
    poss_lbls = tf.reshape(poss_lbls, [35, 35, NUM_CLASSES])
    result.poss_lbls = poss_lbls

    return result




def _generate_image_and_label_batch(image, label, edge, ignore, poss_lbls, min_queue_examples,
                                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
        images, labels, edges, ignore_, poss_lbls_ = tf.train.shuffle_batch(
            [image, label, edge, ignore, poss_lbls],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels, edges, ignore_, poss_lbls_ = tf.train.batch(
            [image, label, edge, ignore, poss_lbls],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels, edges, ignore_, poss_lbls_


def createDS(data_dir = '', batch_size=1):
    # Create image queue
    filenames_ = os.listdir(data_dir + 'test_images/')
    filenames_.sort()
    images = [os.path.join(data_dir + 'test_images/', filenames_[i])
              for i in xrange(0, len(filenames_))]

    # Create label queue
    filenames_ = os.listdir(data_dir + 'test_labels/')
    filenames_.sort()
    labels = [os.path.join(data_dir + 'test_labels/', filenames_[i])
              for i in xrange(0, len(filenames_))]

    # Create edge queue
    filenames_ = os.listdir(data_dir + 'test_edges/')
    filenames_.sort()
    edges = [os.path.join(data_dir + 'test_edges/', filenames_[i])
             for i in xrange(0, len(filenames_))]

    # Create poss_lbls queue
    filenames_ = os.listdir(data_dir + 'test_poss_lbls_/')
    filenames_.sort()
    poss_lbls = [os.path.join(data_dir + 'test_poss_lbls_/', filenames_[i])
                 for i in xrange(0, len(filenames_))]

    queue = tf.train.slice_input_producer([images, labels, edges, poss_lbls])
    # Read examples from files in the filename queue.
    read_input = read_data(queue)
    reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32)  # Changes to range [0,1)
    read_input.label = tf.expand_dims(read_input.label, 2)  # add a third dimension (1 channel)
    read_input.edge = tf.expand_dims(read_input.edge, 2)  # add a third dimension (1 channel)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Crop the central [height, width] of the image.
    reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    read_input.label = tf.image.resize_image_with_crop_or_pad(read_input.label, height, width)
    read_input.edge = tf.image.resize_image_with_crop_or_pad(read_input.edge, height, width)

    # Resize label, edge and ignore to output size
    read_input.label = tf.image.resize_images(read_input.label, [LABEL_SIZE, LABEL_SIZE],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    read_input.edge = tf.image.resize_images(read_input.edge, [LABEL_SIZE, LABEL_SIZE],
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    read_input.poss_lbls = tf.image.resize_images(read_input.poss_lbls, [LABEL_SIZE, LABEL_SIZE],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Also match poss_lbls size

    # Subtract off the mean of the pixels.
    bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
    float_image = (reshaped_image - bsds_mean)

    # Create ignore
    #  read_input.ignore = tf.cast(tf.equal(read_input.label, 0), tf.uint8)
    #  read_input.ignore = tf.concat([read_input.ignore, tf.zeros((LABEL_SIZE, LABEL_SIZE, NUM_CLASSES-1), dtype=tf.uint8)], axis=2)
    read_input.ignore = tf.cast(tf.equal(read_input.label, 0), tf.float32) - tf.cast(tf.not_equal(read_input.label, 0),
                                                                                     tf.float32)
    read_input.ignore = tf.concat(
        [read_input.ignore, tf.zeros((LABEL_SIZE, LABEL_SIZE, NUM_CLASSES - 1), dtype=tf.float32)], axis=2)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label = tf.squeeze(read_input.label)
    read_input.edge = tf.squeeze(read_input.edge)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.3
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, read_input.edge, read_input.ignore,
                                           read_input.poss_lbls,
                                           min_queue_examples, batch_size,
                                           shuffle=False)