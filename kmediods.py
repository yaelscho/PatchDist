from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import shutil
from collections import Counter
from itertools import combinations_with_replacement as comb
import sklearn
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
import pickle
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
from datetime import datetime
import math
import time
from scipy.io import loadmat
import numba
from numba import jit, njit
import os
from PIL import Image
import tensorflow_datasets as tfds
import json
import numpy as np
import tensorflow as tf
from six.moves import xrange
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
from pycocotools.coco import COCO
import multiprocessing
import resnet50_input
import resnet50

FLAGS = tf.compat.v1.app.flags.FLAGS


tf.compat.v1.app.flags.DEFINE_string('eval_dir', 'eval',"""Directory where to write event logs.""")
tf.compat.v1.app.flags.DEFINE_string('train_dir', 'chk',"""Directory where to read model checkpoints.""")
tf.compat.v1.app.flags.DEFINE_boolean('evaluation', False, "Whether this is evaluation or representation calculation")
tf.compat.v1.app.flags.DEFINE_integer('num_examples', 200, """Number of examples to run.""")

def mostCommonClass(patch):
  val, counts = np.unique(patch, return_counts=True)
  ind = np.argmax(counts)
  return val[ind]


def getHist(sess, image, GT, fuse):
  pos_dist = []
  neg_dist = []
  for i in range(FLAGS.num_examples):
    im, gt, rep = sess.run([image, GT, fuse])
    rep = np.squeeze(rep)
    res_im = rep
    res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
    res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
    rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))
    rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 4)
    yVec = np.arange(0, rep.shape[1] - 1, 4)
    for x in xVec:
      for y in yVec:
        # x = np.random.randint(0, rep.shape[1]-1, 2)
        # y = np.random.randint(0, rep.shape[2]-1, 2)
        pix1 = rep[x, y, :]
        randx = np.random.choice(rep.shape[0] - 1)
        randy = np.random.choice(rep.shape[1] - 1)
        # randx = x+5
        # randy = y+5
        pix2 = rep[randx, randy, :]
        dist = np.sqrt(sum(np.power((pix1 - pix2), 2)))

        # GT = loadmat(os.sep.join(['BSDS500/data/groundTruth/test/' + filenames[i] + '.mat']))['groundTruth']
        # person1 = GT[0, 0][0, 0][0]
        # person2 = GT[0, 1][0, 0][0]
        # person3 = GT[0, 2][0, 0][0]
        # person4 = GT[0, 3][0, 0][0]
        #
        x_im = x * 4
        y_im = y * 4
        randx_im = randx * 4
        randy_im = randy * 4
        if x_im == randx_im and y_im == randy_im:
          continue

        score = bool(mostCommonClass(gt, x_im, y_im) == mostCommonClass(gt, randx_im, randy_im))
        if score:
          pos_dist.append(dist)
        else:
          neg_dist.append(dist)
  return pos_dist, neg_dist



def calcPatchDist(patch1, patch2):
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
  dist1 = np.mean([np.amin([np.linalg.norm(pix2-pix1,2) for pix1 in patch1]) for pix2 in patch2])
  dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  return np.mean([dist1, dist2])


@njit(parallel=True)
def calcPatchDistmulti(doues):
  dissim = np.zeros((4224, 4224))
  for dou in tqdm(doues):
    p1, p2 = dou
    patch1 = p1['patch']
    patch2 = p2['patch']
    if len(patch1.shape)>2:
      patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2])
      patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2])
    dist1 = np.mean([np.amin([np.linalg.norm(pix2-pix1,2) for pix1 in patch1]) for pix2 in patch2])
    dist2 = np.mean([np.amin([np.linalg.norm(pix2 - pix1, 2) for pix2 in patch2]) for pix1 in patch1])
  dissim[p1['idx'], p2['idx']] = np.mean([dist1, dist2])
  dissim[p2['idx'], p1['idx']] = np.mean([dist1, dist2])
  return dissim

# @jit(target_backend='cuda')
def calcPatchDisttorch(patch1, patch2):
  if len(patch1.shape)>2:
    patch1 = patch1.reshape(patch1.shape[0]*patch1.shape[1], patch1.shape[2]).to('cuda')
    patch2 = patch2.reshape(patch2.shape[0] * patch2.shape[1], patch2.shape[2]).to('cuda')
  dist1 = torch.mean(torch.Tensor([torch.amin(torch.Tensor([torch.linalg.norm(pix2-pix1,2) for pix1 in patch1])) for pix2 in patch2]))
  dist2 = torch.mean(torch.Tensor([torch.amin(torch.Tensor([torch.linalg.norm(pix2-pix1,2) for pix2 in patch2])) for pix1 in patch1]))
  return torch.mean(torch.Tensor([dist1, dist2]))


def getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=16):
  pos_dist = []
  neg_dist = []
  for i in range(20):
    im, gt, rep = sess.run([image, GT, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0]/2), int(rep.shape[1]/2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size/2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size/2)
        if x_pos<0 or y_pos<0:
          x_pos=x
          y_pos=y
        if x_pos==x and y_pos==y:
          x_pos = x+int(patch_size/2)
        if x < center[0]:
          x_neg = x+ center[0]
        else:
          x_neg = x - center[0]
        if y < center[1]:
          y_neg = y+ center[1]
        else:
          y_neg = y - center[1]
        patch_orig = rep[x:x+patch_size, y:y+patch_size, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]

        pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return pos_dist, neg_dist


def getPatchDS_imagenette(sess, image, labels, fuse, patch_size=4, ifPCA=True, n=16):
  # pos_dist = []
  # neg_dist = []
  patches = []
  for i in range(20):
    im, label, rep = sess.run([image, labels, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 20)
    yVec = np.arange(0, rep.shape[1] - 1, 20)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        if x_pos == x and y_pos == y:
          x_pos = x + int(patch_size / 2)
        if x_pos < 0:
          x_pos = x + int(patch_size / 2)
        if y_pos < 0:
          y_pos = y + int(patch_size / 2)
        if x_pos+patch_size > rep.shape[0]:
          x_pos = x-int(patch_size / 2)
        if y_pos+patch_size > rep.shape[1]:
          y_pos = y-int(patch_size / 2)
        if x < center[0]:
          x_neg = x + (center[0]-patch_size)
        else:
          x_neg = x - (center[0]-patch_size)
        if y < center[1]:
          y_neg = y + (center[1]-patch_size)
        else:
          y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "label": label})
        patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "label": label})
        patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def getPatchDS_COCO(sess, image,mask, labels, fuse, patch_size=4, ifPCA=True, n=16):
  # pos_dist = []
  # neg_dist = []
  dataDir = './COCO'
  annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  coco =COCO(annFile)
  catIDs = coco.getCatIds()
  cats = coco.loadCats(catIDs)
  filterClasses = ['dog', 'elephant', 'fire hydrant', 'train', 'airplane']
  catIds = coco.getCatIds(catNms=filterClasses)
  patches = []
  for i in range(22):
    im, seg, label, rep = sess.run([image, mask, labels, fuse])
    rep = np.squeeze(rep)
    if ifPCA:
      res_vec = rep.reshape((rep.shape[0] * rep.shape[1], rep.shape[2]))
      res_reduced = PCA(n_components=n, svd_solver='full').fit_transform(res_vec)
      rep = np.reshape(res_reduced, (rep.shape[0], rep.shape[1], n))
      # rep_3 = rep_3 - np.amin(rep_3) / np.amax(rep_3 - np.amin(rep_3))
    xVec = np.arange(0, rep.shape[0] - 1, 10)
    yVec = np.arange(0, rep.shape[1] - 1, 10)
    center = int(rep.shape[0] / 2), int(rep.shape[1] / 2)
    for x in xVec:
      for y in yVec:
        x_pos = x + np.random.randint(-1, 1) * int(patch_size / 2)
        y_pos = y + np.random.randint(-1, 1) * int(patch_size / 2)
        if x_pos == x and y_pos == y:
          x_pos = x + int(patch_size / 2)
        if x_pos < 0:
          x_pos = x + int(patch_size / 2)
        if y_pos < 0:
          y_pos = y + int(patch_size / 2)
        if x_pos+patch_size > rep.shape[0]:
          x_pos = x-int(patch_size / 2)
        if y_pos+patch_size > rep.shape[1]:
          y_pos = y-int(patch_size / 2)
        if x < center[0]:
          x_neg = x + (center[0]-patch_size)
        else:
          x_neg = x - (center[0]-patch_size)
        if y < center[1]:
          y_neg = y + (center[1]-patch_size)
        else:
          y_neg = y - (center[1]-patch_size)
        patch_orig = rep[x:x + patch_size, y:y + patch_size, :]
        im_orig = im[0][x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        seg_orig = seg[x * 4: x*4 + patch_size*4, y*4:y* 4 + patch_size *4, :]
        patch_pos = rep[x_pos:x_pos + patch_size, y_pos:y_pos + patch_size, :]
        im_pos = im[0][x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        seg_pos = seg[x_pos * 4: x_pos * 4 + patch_size * 4, y_pos * 4:y_pos * 4 + patch_size * 4, :]
        patch_neg = rep[x_neg:x_neg + patch_size, y_neg:y_neg + patch_size, :]
        im_neg = im[0][x_neg * 4: x_neg*4 + patch_size*4, y_neg*4:y_neg* 4 + patch_size *4, :]
        seg_neg = seg[x_neg * 4: x_neg * 4 + patch_size * 4, y_neg * 4:y_neg * 4 + patch_size * 4, :]
        patches.append({"idx": len(patches), "patch": patch_orig, "ref": im_orig, "seg": mostCommonClass(seg_orig), "label": label})
        patches.append({"idx": len(patches), "patch": patch_pos, "ref": im_pos, "seg": mostCommonClass(seg_pos), "label": label})
        patches.append({"idx": len(patches), "patch": patch_neg, "ref": im_neg, "seg": mostCommonClass(seg_neg), "label": label})
        # pos_dist.append(calcPatchDist(patch_orig, patch_pos))
        # neg_dist.append(calcPatchDist(patch_orig, patch_neg))
  return patches


def preprocess():
  dir = 'imagenette2-320/clustering'
  for f in os.listdir(dir):
    folder = os.path.join(dir, f)
    for im in os.listdir(folder):
      orig = plt.imread(os.path.join(folder, im))
      if orig.shape[1]>orig.shape[0]:
        s = int(orig.shape[1]/2)-160
        cropped = orig[:,s:s+320,:]
      else:
        s = int(orig.shape[0] / 2) - 160
        cropped = orig[s:s + 320, :, :]
      plt.imsave(os.path.join(folder, im), cropped)


def getClassName(classID, cats):
  for i in range(len(cats)):
    if cats[i]['id'] == classID:
      return cats[i]['name']
  return "None"


def preprocess_COCO(imgNames, coco):
  dir = 'COCO/clustering'
  if os.path.isdir(dir):
    shutil.rmtree(dir)
  os.mkdir(dir)
  dir_mask = 'COCO/clustering/mask'
  os.mkdir(dir_mask)
  catIDs = coco.getCatIds()
  cats = coco.loadCats(catIDs)
  filterClasses = ['dog', 'elephant', 'fire hydrant', 'train', 'airplane']
  catIds = coco.getCatIds(catNms=filterClasses)
  images = []
  labels = []
  masks = []
  minh = np.inf
  minw = np.inf
  for im in imgNames:
    # shutil.copy('COCO/train2017/{}'.format(im), os.path.join(dir, im))
    # img = plt.imread(os.path.join(dir, im))
    # print(img.shape)
    imgId = int(im.split('.')[0])
    img = coco.loadImgs(imgId)[0]
    if img['height']< minh:
      minh = img['height']
    if img['width'] < minw:
      minw = img['width']
  # shape = min(minh,minw)
  shape = 320
  for im in imgNames:
    orig = Image.open('COCO/train2017/{}'.format(im))
    imrsz = np.array(orig)[:shape, :shape, :]
    plt.imsave(os.path.join(dir, im), imrsz)
    # img = plt.imread(os.path.join(dir, im))
    # print(img.shape)
    imgId = int(im.split('.')[0])
    img = coco.loadImgs(imgId)[0]
    mask = np.zeros((img['height'], img['width']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for i in range(len(anns)):
      className = getClassName(anns[i]['category_id'], cats)
      pixel_value = filterClasses.index(className) + 1
      mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
    # mask = Image.fromarray(mask).resize((shape, shape))
    # mask = cv2.resize(mask, dsize=(shape, shape), interpolation=cv2.INTER_NEAREST)
    plt.imsave(os.path.join(dir_mask,im), mask[:320,:320], cmap='gray')
    images.append(os.path.join(dir, im))
    masks.append(os.path.join(dir_mask,im))
    labels.append(className)
  return images, masks, labels
    # plt.imshow(mask)
    # folder = os.path.join(dir, f)
    # for im in os.listdir(folder):
    #
    #   if orig.shape[1]>orig.shape[0]:
    #     s = int(orig.shape[1]/2)-160
    #     cropped = orig[:,s:s+320,:]
    #   else:
    #     s = int(orig.shape[0] / 2) - 160
    #     cropped = orig[s:s + 320, :, :]
    #   plt.imsave(os.path.join(folder, im), cropped)

def getRep_imagenette():
  # preprocess()
  dir = 'imagenette2-320/clustering'
  folders = os.listdir(dir)
  images = []
  labels = []
  for l in folders:
    path = os.path.join(dir, l)
    filenames_ = os.listdir(path)
    filenames_.sort()
    img_lst = [os.path.join(dir, l, filenames_[i])
              for i in xrange(0, len(filenames_))]
    images.extend(img_lst)
    labels.extend([l for i in img_lst])

  queue = tf.compat.v1.train.slice_input_producer([images, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  label = queue[1]
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  image.set_shape([320, 320, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)  # Changes to range [0,1)
  image = tf.expand_dims(image, 0)

  return image, label

def getRep_coco():
  dataDir = './COCO'
  annFile = '{}/annotations/instances_train2017.json'.format(dataDir)
  coco =COCO(annFile)
  with open('COCO/imgNames.pkl', 'rb') as f:
    imgNames = pickle.load(f)
  images, masks, labels = preprocess_COCO(imgNames, coco)

  # dir = 'imagenette2-320/clustering'
  # folders = os.listdir(dir)
  # images = []
  # labels = []
  # for l in folders:
  #   path = os.path.join(dir, l)
  #   filenames_ = os.listdir(path)
  #   filenames_.sort()
  #   img_lst = [os.path.join(dir, l, filenames_[i])
  #             for i in xrange(0, len(filenames_))]
  #   images.extend(img_lst)
  #   labels.extend([l for i in img_lst])

  queue = tf.compat.v1.train.slice_input_producer([images, masks, labels], num_epochs=600, shuffle=False)
  image = tf.io.read_file(queue[0])
  mask = tf.io.read_file(queue[1])
  label = queue[2]
  mask = tf.image.decode_jpeg(mask, channels=1)
  image = tf.image.decode_jpeg(image, channels=3)
  # shape = tf.shape(image)
  image.set_shape([320, 320, 3])
  mask.set_shape([320, 320, 1])
  image = tf.image.convert_image_dtype(image, tf.float32)  # Changes to range [0,1)
  image = tf.expand_dims(image, 0)

  return image, mask, label


@jit(parallel=True)
def calcdissim(patches):
  dissim = np.zeros((len(patches), len(patches)))
  for i in range(len(patches)):
    for j in tqdm(range(len(patches))):
      p1 = patches[i]
      p2 = patches[j]
      if dissim[p1['idx'], p2['idx']] != 0:
        continue
      else:
        # start = time.time()
        # patch1 = torch.from_numpy(p1['patch'])
        # patch2 = torch.from_numpy(p2['patch'])
        # dist = calcPatchDisttorch(patch1, patch2)
        dist = calcPatchDist(p1['patch'], p2['patch'])
        dissim[p1['idx'], p2['idx']] = dist
        dissim[p2['idx'], p1['idx']] = dist
  return dissim


def get_representations():
  imagenette = False
  coco = True
  if not imagenette and not coco:
    image, GT, filenames = resnet50_input.test_inputs(FLAGS.data_dir)
    filenames = [element.split('.')[0] for element in filenames]
    # Build a Graph that computes the logits predictions from the
    # inference model.
    _, _, fuse = resnet50.inference(image, tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]), tf.zeros([1,image.shape[1]//4+image.shape[1]%4,image.shape[2]//4+image.shape[2]%4,resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
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

        dir = 'BSDS500/data/images/test'
        dirlist = sorted(os.listdir(dir))

        # pos_dist, neg_dist = getHist(sess, image, GT, fuse)
        # pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=8, ifPCA=True, n=6)
        # pos_d_pca = []
        # neg_d_pca = []
        # for i in range(1,9):
        #   pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=2**i)
        #   pos_d_pca.append([pos_dist])
        #   neg_d_pca.append([neg_dist])
        pos_dist, neg_dist = getPatchDS(sess, image, GT, fuse, patch_size=4, ifPCA=True, n=16)
        # pos_d_pca.append([pos_dist])
        # neg_d_pca.append([neg_dist])

        # for i in range(FLAGS.num_examples):
        #   im, gt, rep = sess.run([image, GT, fuse])
        #   rep = np.squeeze(rep)
        #   res_im = np.squeeze(rep)
        #   res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
        #   res_reduced = PCA(n_components=3, svd_solver='full').fit_transform(res_vec)
        #   rep_3 = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 3))
        #   rep_3 = rep_3- np.amin(rep_3)/np.amax(rep_3- np.amin(rep_3))
        #   xVec = np.arange(0, rep.shape[0]-1, 4)
        #   yVec = np.arange(0, rep.shape[1]-1, 4)
        #
        #   # ax2.imshow(im.squeeze())
        #   # file_name = ''
        #   # for img in dirlist:
        #   #   if img.split('.')[1] == 'jpg':
        #   #     # img_read = plt.imread(dir + '/' + img)/255
        #   #     # img_read -= np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        #   #     img_read = tf.io.read_file(dir + '/' + img)
        #   #     # image = tf.read_file(queue[0])
        #   #     img_read = tf.image.decode_jpeg(img_read, channels=3)
        #   #     try:
        #   #       img_read.set_shape([321, 481, 3])
        #   #     except:
        #   #       img_read.set_shape([481, 321, 3])
        #   #     img_read = tf.image.convert_image_dtype(img_read, tf.float32)
        #   #     # bsds_mean = np.array([0.4342, 0.4428, 0.3673], dtype=np.float32)
        #   #     # img_read = img_read - bsds_mean
        #   #     img_read = sess.run(img_read)
        #   #     if im.squeeze().shape==img_read.shape:
        #   #       if np.sum(img_read-im.squeeze())==0:
        #   #         file_name = img.split('.')[0]
        #   #         dirlist.remove(img)
        #   #         break
        #   # if not file_name:
        #   #   continue
        #   for x in xVec:
        #     for y in yVec:
        #       # x = np.random.randint(0, rep.shape[1]-1, 2)
        #       # y = np.random.randint(0, rep.shape[2]-1, 2)
        #       pix1 = rep[x, y, :]
        #       randx = np.random.choice(rep.shape[0]-1)
        #       randy = np.random.choice(rep.shape[1]-1)
        #       # randx = x+5
        #       # randy = y+5
        #       pix2 = rep[randx, randy, :]
        #       dist = np.sqrt(sum(np.power((pix1 - pix2), 2)))
        #
        #       # GT = loadmat(os.sep.join(['BSDS500/data/groundTruth/test/' + filenames[i] + '.mat']))['groundTruth']
        #       # person1 = GT[0, 0][0, 0][0]
        #       # person2 = GT[0, 1][0, 0][0]
        #       # person3 = GT[0, 2][0, 0][0]
        #       # person4 = GT[0, 3][0, 0][0]
        #       #
        #       x_im = x * 4
        #       y_im = y * 4
        #       randx_im = randx * 4
        #       randy_im = randy * 4
        #       if x_im==randx_im and y_im==randy_im:
        #         continue
        #
        #       score = bool(mostCommonClass(gt, x_im, y_im) == mostCommonClass(gt, randx_im, randy_im))
        #       # p1_score = int(mostCommonClass(person1, x_im, y_im) == mostCommonClass(person1, randx_im, randy_im))
        #       # p2_score = int(mostCommonClass(person2, x_im, y_im) == mostCommonClass(person2, randx_im, randy_im))
        #       # p3_score = int(mostCommonClass(person3, x_im, y_im) == mostCommonClass(person3, randx_im, randy_im))
        #       # p4_score = int(mostCommonClass(person4, x_im, y_im) == mostCommonClass(person4, randx_im, randy_im))
        #       # if GT.shape[1]==5:
        #       #   person5 = GT[0, 4][0, 0][0]
        #       #   p5_score = int(mostCommonClass(person5, x_im, y_im) == mostCommonClass(person5, randx_im, randy_im))
        #       #   tot_score = sum((p1_score, p2_score, p3_score, p4_score, p5_score)) / 5
        #       # else:
        #       #   tot_score = sum((p1_score, p2_score, p3_score, p4_score)) / 4
        #       if score:
        #         pos_dist.append(dist)
        #       else:
        #         neg_dist.append(dist)
        #       # if tot_score>0.7:
        #       #   pos_dist.append(dist)
        #       # if tot_score<0.3:
        #       #   neg_dist.append(dist)
        #       # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #       # ax1.imshow(rep_3/np.amax(rep_3))
        #       # ax3.imshow(person1)
        #       # ax4.imshow(person2)
        #       # ax2.imshow(person3)
        #       # plt.show()
        #       # plt.close()
        #       # plt.imshow(rep_3)
        #       # plt.scatter([y_im, randy_im], [x_im, randx_im], s=30, marker='o', c='r')
        #       # plt.show()
        #       # a=1

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    plt.hist(pos_dist, bins=50)
    plt.hist(neg_dist, bins=50, alpha=0.3)
    plt.show()
  elif imagenette:
    image, label = getRep_imagenette()
    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
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
        patches = getPatchDS_imagenette(sess, image, label, fuse, patch_size=8, ifPCA=True, n=16)

        # with open('patches_8.pkl', 'rb') as f:
        #   patches = pickle.load(f)
        # patches = random.choices(patches, k=1000)
        dissim = np.zeros((len(patches), len(patches)))
        for p1 in tqdm(patches):
            for p2 in patches:
              if dissim[p1['idx'], p2['idx']] != 0:
                continue
              else:
                # start = time.time()
                # patch1 = torch.from_numpy(p1['patch'])
                # patch2 = torch.from_numpy(p2['patch'])
                # dist = calcPatchDisttorch(patch1, patch2)
                dist = calcPatchDist(p1['patch'], p2['patch'])
                dissim[p1['idx'], p2['idx']] = dist
                dissim[p2['idx'], p1['idx']] = dist
                # print(time.time()-start)
        max_d = np.amax(dissim)
        affinity = 1-(dissim/max_d)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i,i] = sum(affinity[i,:])
        L = np.matmul(np.matmul(np.diag(np.diag(D)**(-1/2)), affinity), np.diag(np.diag(D)**(-1/2)))
        eig = np.linalg.eig(L)
        space = eig[1][:, :10]
        normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
        Y = (space.T / normalize_f).T
        patches = sorted(patches, key=lambda p: p['idx'])
        mymetric = sklearn.metrics.make_scorer(calcPatchDist)
        clustering_kmeans = SpectralClustering(n_clusters=50,
                                        assign_labels='kmeans',
                                        random_state=0,
                                        affinity='nearest_neighbors').fit(np.array([p['patch'].flatten() for p in patches]))
        clustering_disc = SpectralClustering(n_clusters=50,
                                               assign_labels='discretize',
                                               random_state=0,
                                               affinity='nearest_neighbors').fit(
          np.array([p['patch'].flatten() for p in patches]))

        plt.hist(clustering_disc.labels_, bins=50)
        plt.hist(clustering_kmeans.labels_, bins=50)
        a =1




      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  elif coco:
    image,mask, label = getRep_coco()
    _, _, fuse = resnet50.inference(image, tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]), tf.zeros(
      [1, image.shape[1] // 4 + image.shape[1] % 4, image.shape[2] // 4 + image.shape[2] % 4,
       resnet50_input.NUM_CLASSES]))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
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
        # patches = getPatchDS_COCO(sess, image, mask, label, fuse, patch_size=8, ifPCA=True, n=16)
        #
        # with open('patches_8_coco.pkl', 'wb') as f:
        #   pickle.dump(patches, f)
        with open('patches_8_coco.pkl', 'rb') as f:
          patches = pickle.load(f)
        # patches = random.choices(patches, k=500)

        doues = comb(patches, 2)
        dissim = calcPatchDistmulti(doues)
        # with multiprocessing.Pool(processes=16) as p:
        #   with tqdm(total=(4224 ** 2)/2) as pbar:
        #     for d in p.imap_unordered(calcPatchDistmulti, doues):
        #       pbar.update()
        #       dissim[d[0], d[1]] = d[2]
        #       dissim[d[1], d[0]] = d[2]
        # # for p1 in tqdm(patches):
        # #   for p2 in patches:
        # #     if dissim[p1['idx'], p2['idx']] != 0:
        # #       continue
        # #     else:
        # #       # start = time.time()
        # #       # patch1 = torch.from_numpy(p1['patch'])
        # #       # patch2 = torch.from_numpy(p2['patch'])
        # #       # dist = calcPatchDisttorch(patch1, patch2)
        # #       dist = calcPatchDist(p1['patch'], p2['patch'])
        # #       dissim[p1['idx'], p2['idx']] = dist
        # #       dissim[p2['idx'], p1['idx']] = dist
        # #       # print(time.time()-start)
        with open('dissim_8_coco_{}.pkl'.format(datetime.now()), 'wb') as f:
          pickle.dump(dissim, f)
        # with open('dissim_8_coco.pkl', 'rb') as f:
        #   dissim = pickle.load(f)
        max_d = np.amax(dissim)
        affinity = 1 - (dissim / max_d)
        sigma = np.mean(dissim)
        affinity = np.exp(-dissim/sigma)
        D = np.zeros(affinity.shape)
        for i in range(affinity.shape[0]):
          D[i, i] = sum(affinity[i, :])
        L = np.matmul(np.matmul(np.diag(np.diag(D) ** (-1 / 2)), affinity), np.diag(np.diag(D) ** (-1 / 2)))
        eig = np.linalg.eig(L)
        # vLens = range(6, 51, 2)
        vLens = [40]
        clustersCenterD = []
        inClustersD = []
        ns = range(10,51,5)
        # ns = [40]
        fig2, ax2 = plt.subplots()
        for n in ns:
          for vLen in vLens:
            space = eig[1][:, :vLen]
            normalize_f = np.sqrt((np.sum(np.square(space), axis=1)))
            Y = (space.T / normalize_f).T
            Kmeans = sklearn.cluster.KMeans(n_clusters=n, algorithm='elkan')
            clusters = Kmeans.fit(Y)
            clusters_trans = Kmeans.fit_transform(Y)
            clustersCenterD.append(np.mean(
              [[np.linalg.norm((clusters.cluster_centers_[i, :]-clusters.cluster_centers_[j, :]), 2) for i in range(n)]
               for j in range(n) if i != j]))
            inClustersD.append(np.mean([np.linalg.norm((Y[i]-clusters.cluster_centers_[clusters.labels_[i]]),2) for i in range(len(patches))]))
        # plot the change in the distances according to the size of the new patches descriptors
        #   ax2.plot(vLens, clustersCenterD[-len(vLens):], label='cluster centers mean distance n={}'.format(n))
        #   ax2.plot(vLens, inClustersD[-len(vLens):], label='in-clusters mean distance n={}'.format(n))
        #   ax2.set_xlabel('# eig vec')
        #   ax2.set_ylabel('mean distance')
        #   ax2.legend()

        # simulate and test the population of one cluster
          population = np.zeros((n, 10)) # 0- Airplane 1- Airplane backgrouns 2- dog 3- dog background 4- elephant /
          # 5- elephant background 6- fire hydrant 7- fire hydrant background 8- train 9- train background
          clustersNames = range(n)
          labelsNames = ['Airplane', 'Airplane backgrouns', 'dog', 'dog background','elephant',
          'elephant background', 'fire hydrant', 'fire hydrant background', 'train', 'train background']
          for clusterIDX in range(n):
            idxlist = np.where(clusters.labels_ == clusterIDX)
            print(len(idxlist)/len(patches))
            GT_cluster = [(p['label'], p['seg']) for p in patches if p['idx'] in idxlist[0]]
            classesPop = np.unique(GT_cluster, return_counts=True, axis=0)
            for i, pop in enumerate(classesPop[0]):
              if b'airplane' in pop and b'255' in pop:
                classesPop[0][i] = 0
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'airplane' in pop and b'0' in pop:
                classesPop[0][i] = 1
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'dog' in pop and b'255' in pop:
                classesPop[0][i] = 2
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'dog' in pop and b'0' in pop:
                classesPop[0][i] = 3
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'fire hydrant' in pop and b'255' in pop:
                classesPop[0][i] = 6
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'fire hydrant' in pop and b'0' in pop:
                classesPop[0][i] = 7
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'elephant' in pop and b'255' in pop:
                classesPop[0][i] = 4
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'elephant' in pop and b'0' in pop:
                classesPop[0][i] = 5
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'train' in pop and b'255' in pop:
                classesPop[0][i] = 8
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
              if b'train' in pop and b'0' in pop:
                classesPop[0][i] = 9
                population[clusterIDX, int(classesPop[0][i][0])] = classesPop[1][i]
          fig, ax = plt.subplots()
          im = ax.imshow(population)
          ax.set_yticks(np.arange(len(clustersNames)))
          ax.set_xticks(np.arange(len(labelsNames)))
          cb = plt.colorbar(im)
          plt.savefig('heatmaps_coco_clustering/popHeatmap_{}_clusters_{}_eigvecs.png'.format(n, vLen))
          plt.close(fig)
          # population[clusterIDX, classesPop[0]] = classesPop[1]
          # plt.pie(np.unique(GT_cluster, return_counts=True, axis=0)[1],
          #         labels=np.unique(GT_cluster, return_counts=True, axis=0)[0])
        # plt.show()
        # plt.savefig('keepDaway.png')
        # plt.close(fig2)
        fig, ax = plt.subplots()
        ax.imshow(population)
        ax.set_yticks(np.arange(len(clustersNames)), labels=clustersNames)
        ax.set_xticks(np.arange(len(labelsNames)), labels=labelsNames)
        ims = [p['ref'] for p in patches if p['idx'] in idxlist[0]]
        random.shuffle(ims)
        fig, axs = plt.subplots(6, 6)
        for i in range(6):
          for j in range(6):
            axs[i, j].imshow(ims[i * 6 + j])
        plt.show()
        # patches = sorted(patches, key=lambda p: p['idx'])
        # mymetric = sklearn.metrics.make_scorer(calcPatchDist)
        # clustering_kmeans = SpectralClustering(n_clusters=50,
        #                                        assign_labels='kmeans',
        #                                        random_state=0,
        #                                        affinity='nearest_neighbors').fit(
        #   np.array([p['patch'].flatten() for p in patches]))
        # clustering_disc = SpectralClustering(n_clusters=50,
        #                                      assign_labels='discretize',
        #                                      random_state=0,
        #                                      affinity='nearest_neighbors').fit(
        #   np.array([p['patch'].flatten() for p in patches]))
        #
        # plt.hist(clustering_disc.labels_, bins=50)
        # plt.hist(clustering_kmeans.labels_, bins=50)


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
