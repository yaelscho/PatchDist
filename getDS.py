import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.io import loadmat
from six.moves import xrange

def read_data(queue):
    class DATARecord(object):
        pass

    result = DATARecord()

    image = tf.read_file(queue[0])
    result.uint8image = tf.image.decode_jpeg(image, channels=3)
    shape = tf.shape(result.uint8image)

    label = tf.read_file(queue[1])
    label = tf.image.decode_jpeg(label, channels=1)
    # label = tf.reshape(label, [shape[1], shape[0]])
    # label = tf.transpose(label)
    result.label = label
    return result


def createDS():
    # img_list = []
    # GT_Seg = []
    # GT_B = []
    # GTs = []
    filenames_ = os.listdir('BSDS500/data/images/test/')
    filenames_.sort()
    images = [os.path.join('BSDS500/data/images/test/', filenames_[i])
              for i in xrange(0, len(filenames_))]

    # Create label queue
    filenames_ = os.listdir('BSDS500/data/groundTruth/test/')
    filenames_ = [item for item in filenames_ if item.endswith('.jpg')]
    filenames_.sort()
    labels = [os.path.join('BSDS500/data/groundTruth/test/', filenames_[i])
              for i in xrange(0, len(filenames_))]

    queue = tf.train.slice_input_producer([images, labels])
    read_input = read_data(queue)
    image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32)  # Changes to range [0,1)
    # read_input.label = tf.expand_dims(read_input.label, 2)
    image.set_shape([321,481,3])
    label = read_input.label
    label.set_shape([321, 481, 1])
    return tf.train.batch([image, label], batch_size=1, num_threads=1)
    # images, labels = tf.train.batch([image, label], batch_size=1, num_threads=1)
    # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images,labels])
    #     filename = im.split('.')[0]
    #     img_list.append(tf.convert_to_tensor(np.expand_dims((plt.imread('BSDS500/data/images/test/'+im)/255).astype(np.float32),0)))
    #     GT = loadmat('BSDS500/data/groundTruth/test/'+filename+'.mat')['groundTruth'][0]
    #     # GT_Seg.append(np.array([GT[i][0][0][0] for i in range(4)]))
    #     # GT_B.append(np.array([GT[i][0][0][1] for i in range(4)]))
    #     GTs.append(tf.convert_to_tensor(GT[0][0][0][0]))
    #
    #
    # # GT_Seg = sorted(['BSDS500/data/groundTruth/test/'+ item for item in os.listdir('BSDS500/data/groundTruth/test/') if '_seg_' in item])
    # # GT_B = sorted(['BSDS500/data/groundTruth/test/'+ item for item in os.listdir('BSDS500/data/groundTruth/test/') if '_b_' in item])
    #
    # GT_Seg = tf.convert_to_tensor(GT_Seg)
    # GT_B = tf.convert_to_tensor(np.array(GT_B))
    # GTs = tf.convert_to_tensor(GTs)
    # imgs = tf.convert_to_tensor(img_list)

    # GT = sorted(os.listdir('BSDS500/data/groundTruth/test/'))
    # imgs = sorted(os.listdir('BSDS500/data/images/test/'))

    # filenamesQueue = tf.compat.v1.train.slice_input_producer([sorted([im.split('.')[0] for im in os.listdir('BSDS500/data/images/test/')])])
    # queue = tf.compat.v1.train.slice_input_producer(([imgs, GT_Seg, GT_B]), shuffle=False, num_epochs=600)
    # queue = tf.data.Dataset.from_tensor_slices((imgs, GT_Seg, GT_B))
    image, gt = tf.train.slice_input_producer([imgs, GTs], batch_size=1, num_threads=1)
    queue = tf.contrib.slim.prefetch_queue.prefetch_queue([image, gt])

    return queue

# createDS()
# def convertToJpeg():
#     GT = os.listdir('BSDS500/data/groundTruth/test/')
#     for mat in GT:
#         temp = loadmat('BSDS500/data/groundTruth/test/'+mat)['groundTruth'][0]
#         gt_seg = temp[0][0][0][0]
#         plt.imsave('BSDS500/data/groundTruth/test/'+ mat.split('.')[0]+'.jpg', gt_seg)
#

# convertToJpeg()

