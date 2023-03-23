from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from datetime import datetime
import math
import time
from scipy.io import loadmat
import os

import numpy as np
import tensorflow as tf
from getDS2 import *
import resnet50_input
import resnet50

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string('eval_dir', 'eval',"""Directory where to write event logs.""")
tf.compat.v1.app.flags.DEFINE_string('train_dir', 'chk',"""Directory where to read model checkpoints.""")
tf.compat.v1.app.flags.DEFINE_boolean('evaluation', True, "Whether this is evaluation or representation calculation")
tf.compat.v1.app.flags.DEFINE_integer('num_examples', 200, """Number of examples to run.""")

def mostCommonClass(person, x, y):
  val, counts = np.unique(person[x:x+4, y:y+4], return_counts=True)
  ind = np.argmax(counts)
  return val[ind]


def get_representations():
  # image, filenames = resnet50.test_inputs()
  # filenames = [element.split('.')[0] for element in filenames]
  # Build a Graph that computes the logits predictions from the
  # inference model.

  image, GT = createDS()
  # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([image, GT])
  # iter = queue.make_one_shot_iterator()
  # el = iter.get_next()
  # image, GTseg, GTb = el
  _, _, fuse = resnet50.inference(image, tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]), tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]))
  saver = tf.train.Saver(tf.global_variables())
  with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                             intra_op_parallelism_threads=1)) as sess:
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=False,
                                         start=True))
      pos_dist = []
      neg_dist = []
      dir = 'BSDS500/data/images/test'
      dirlist = sorted(os.listdir(dir))
      for i in range(FLAGS.num_examples):
        # image, GT =  batch_queue.dequeue()
        rep = sess.run(fuse)
        rep = np.squeeze(rep)/np.amax(rep)
        res_im = np.squeeze(rep)
        res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
        res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
        rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))

        xVec = np.arange(0, rep.shape[0]-1, 4)
        yVec = np.arange(0, rep.shape[1]-1, 4)
        im = sess.run(image)
        # ax2.imshow(im.squeeze())
        # file_name = ''
        # for img in dirlist:
        #   if img.split('.')[1] == 'jpg':
        #     # img_read = plt.imread(dir + '/' + img)/255
        #     # img_read -= np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        #     img_read = tf.io.read_file(dir + '/' + img)
        #     # image = tf.read_file(queue[0])
        #     img_read = tf.image.decode_jpeg(img_read, channels=3)
        #     try:
        #       img_read.set_shape([321, 481, 3])
        #     except:
        #       img_read.set_shape([481, 321, 3])
        #     img_read = tf.image.convert_image_dtype(img_read, tf.float32)
        #     # bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        #     # img_read = img_read - bsds_mean
        #     img_read = sess.run(img_read)
        #     if im.squeeze().shape==img_read.shape:
        #       if np.sum(img_read-im.squeeze())==0:
        #         file_name = img.split('.')[0]
        #         dirlist.remove(img)
        #         break
        # if not file_name:
        #   continue
        for x in xVec:
          for y in yVec:
            # x = np.random.randint(0, rep.shape[1]-1, 2)
            # y = np.random.randint(0, rep.shape[2]-1, 2)
            pix1 = rep[x, y, :]
            randx = np.random.choice(rep.shape[0]-1)
            randy = np.random.choice(rep.shape[1]-1)
            # randx = x+5
            # randy = y+5
            pix2 = rep[randx, randy, :]
            dist = np.sqrt(sum(np.power((pix1 - pix2), 2)))

            GT = loadmat(os.sep.join(['BSDS500/data/groundTruth/test/' + filenames[i] + '.mat']))['groundTruth']
            person1 = GT[0, 0][0, 0][0]
            person2 = GT[0, 1][0, 0][0]
            person3 = GT[0, 2][0, 0][0]
            person4 = GT[0, 3][0, 0][0]

            x_im = x * 4
            y_im = y * 4
            randx_im = randx * 4
            randy_im = randy * 4
            if x_im==randx_im and y_im==randy_im:
              continue
            p1_score = int(mostCommonClass(person1, x_im, y_im) == mostCommonClass(person1, randx_im, randy_im))
            p2_score = int(mostCommonClass(person2, x_im, y_im) == mostCommonClass(person2, randx_im, randy_im))
            p3_score = int(mostCommonClass(person3, x_im, y_im) == mostCommonClass(person3, randx_im, randy_im))
            p4_score = int(mostCommonClass(person4, x_im, y_im) == mostCommonClass(person4, randx_im, randy_im))
            if GT.shape[1]==5:
              person5 = GT[0, 4][0, 0][0]
              p5_score = int(mostCommonClass(person5, x_im, y_im) == mostCommonClass(person5, randx_im, randy_im))
              tot_score = sum((p1_score, p2_score, p3_score, p4_score, p5_score)) / 5
            else:
              tot_score = sum((p1_score, p2_score, p3_score, p4_score)) / 4
            if tot_score>0.7:
              pos_dist.append(dist)
            if tot_score<0.3:
              neg_dist.append(dist)
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # ax1.imshow(rep_3/np.amax(rep_3))
            # ax3.imshow(person1)
            # ax4.imshow(person2)
            # ax2.imshow(person3)
            # plt.show()
            # plt.close()
            im = sess.run(image)
            # plt.imshow(rep_3)
            # plt.scatter([y_im, randy_im], [x_im, randx_im], s=30, marker='o', c='r')
            # plt.show()
            # a=1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  plt.hist(pos_dist, bins=50)
  plt.show()
  plt.hist(neg_dist, bins=50, alpha=0.3)
  plt.show()


def eval_once(saver, summary_writer, top_k_op, summary_op, ignore, preds):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))


      num_iter = int(math.floor(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      zeros_count = 0
      step = 0
      while step < num_iter:
        predictions, ignore_, preds_ = sess.run([top_k_op, ignore, preds])
        true_count += np.sum(predictions)
        zeros_count += np.sum(ignore_)
        step += 1

      # Compute precision @ 1.
      precision = (true_count - zeros_count) / (num_iter*FLAGS.batch_size*resnet50.LABEL_SIZE*resnet50.LABEL_SIZE - zeros_count)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return precision


def evaluate():
  with tf.Graph().as_default() as g:
    # images, labels, _, ignore, poss_lbls = createDS()
    #
    # # Build a Graph that computes the logits predictions from the
    # # inference model.
    # logits, ignore_, fuse = resnet50.inference(images, ignore, poss_lbls)
    #
    # gv = tf.global_variables()
    # saver = tf.train.Saver(tf.global_variables())
    # summary_op = tf.summary.merge_all()
    #
    # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    with tf.Session() as sess:
      images, labels, _, ignore, poss_lbls = createDS()
      logits, ignore_, fuse = resnet50.inference(images, ignore, poss_lbls)
      saver = tf.train.Saver(tf.global_variables())
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))
        for i in range(FLAGS.num_examples):
          # image, GT =  batch_queue.dequeue()
          rep, im, gt = sess.run(fuse, images, labels)
          rep = np.squeeze(rep)/np.amax(rep)
          res_im = np.squeeze(rep)
          res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
          res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
          rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument-+++++++++++++++++++++
  if tf.io.gfile.exists(FLAGS.eval_dir):
    tf.compat.v1.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.io.gfile.makedirs(FLAGS.eval_dir)
  if FLAGS.evaluation:
    evaluate()
  else:
    get_representations()




if __name__ == '__main__':
  tf.compat.v1.app.run()
