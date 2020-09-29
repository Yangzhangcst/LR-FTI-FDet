#!/usr/bin/env python

# -*- coding:utf-8 -*-

import _init_paths 
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
import pickle
import cv2
from fast_rcnn.config import cfg
from fast_rcnn.test import _get_blobs
from fast_rcnn.nms_wrapper import nms, soft_nms
import caffe
import scipy.io as sio

deployPrototxt = '../models/pascal_voc/VGG16/faster_rcnn_end2end/test_dsf9.prototxt'
modelFile = '/home/yzhang/soft-nms-3-2/Result-MRPN9-MLPS9-2048/output-BBK/faster_rcnn_end2end/vgg16_faster_rcnn_iter_70000.caffemodel'

#imageListFile = '/home/chenjie/DataSet/CompCars/data/train_test_split/classification/test_model431_label_start0.txt'
#imageBasePath = '/home/chenjie/DataSet/CompCars/data/cropped_image'
#resultFile = 'PredictResult.txt'
 
def initilize():
    print 'initilize ... '
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net
 

def getNetDetails(image, net, boxes=None):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    img = cv2.imread(image)
    rows, cols, channels = img.shape
    M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 0, 1)
    im = cv2.warpAffine(img, M_rotate, (cols, rows))
    blobs, im_scales = _get_blobs(im, boxes)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    #filters = net.params['conv1'][0].data
    #with open('FirstLayerFilter.pickle','wb') as f:
    #   pickle.dump(filters,f)
    #vis_square(filters.transpose(0, 2, 3, 1))

    #feat = net.blobs['conv1'].data[0,:64]
    #with open('FirstLayerOutput.pickle','wb') as f:
    #   pickle.dump(feat,f)
    #vis_square(feat,padval=1)
    pool = net.blobs['fire9/expand3x3'].data[0,:64]  
    with open('pool1.pickle','wb') as f:
       pickle.dump(pool,f)
    vis_square(pool,padval=1)
    #pool2 = net.blobs['pool3'].data[0,:64]
    #with open('pool3.pickle','wb') as f:
    #   pickle.dump(pool2,f)
    #vis_square(pool2,padval=1)
    #pool3 = net.blobs['pool5'].data[0,:128]
    #with open('pool5.pickle','wb') as f:
    #   pickle.dump(pool3,f)
    #vis_square(pool3,padval=1)  
 
def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()
    

    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = ((0, n**2-data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    t_data = 0
    for ind in range(len(data)):
       t_data += data[ind]
    m_data = t_data / len(data) 
    plt.imshow(m_data)
    plt.show()
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data)
    plt.show()
 
if __name__ == "__main__":
   net = initilize()
   cfg.TEST.HAS_RPN = True 
   testimage = '/home/yzhang/caffe-ssd-s/tfdsSqueezeNetV2-SSD/000003.jpg'    # Your test picture path
   getNetDetails(testimage, net)
