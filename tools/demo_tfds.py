#!/usr/bin/env python2
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms, soft_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__','target','fault')

def outputDetectionResult(im, class_name, dets, thresh=0.5): 
    outputFile = open('DetectionResult.txt')
    inds = np.where(dets[:,-1] >= thresh)[0]
    if len(inds) == 0:
        return

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, basePath, testFileName, classes):
    """Detect object classes in an image using pre-computed object proposals."""
    ftest = open(testFileName,'r')
    imageFileName = basePath+'/'+ftest.readline().strip()
    num = 1
    outputFile = open('DetectionResult.txt','w')
    while imageFileName:
     print (imageFileName)
     print 'now is ', num
     num +=1
     imageFileBaseName = os.path.basename(imageFileName)
     imageFileDir = os.path.dirname(imageFileName)

    # Load the demo image
     img = cv2.imread(imageFileName)
     rows, cols, channels = img.shape
     M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 0, 1)
     im = cv2.warpAffine(img, M_rotate, (cols, rows))

    # Detect all object classes and regress object bounds
     timer = Timer()
     timer.tic()
     scores, boxes = im_detect(net, im)
     timer.toc()
     print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
     CONF_THRESH = 0.8
     NMS_THRESH = 0.3
     for cls in classes:
             cls_ind = CLASSES.index(cls)
             cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
             cls_scores = scores[:, cls_ind]
             dets = np.hstack((cls_boxes,
                           cls_scores[:, np.newaxis])).astype(np.float32)
             keep = nms(dets, NMS_THRESH)
             #keep = soft_nms(dets, 0.5, 0.3, 0.001, 1)
             dets = dets[keep, :]
             print 'All {} detections with p({} | box) >= {:.2f}'.format(cls, cls, CONF_THRESH)
     inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
     print 'inds.size', inds.size
     if len(inds) != 0:
         outputFile.write(imageFileName+' ')
         outputFile.write(str(inds.size)+' ')
         for i in inds:
	      	 bbox = dets[i, :4]
         outputFile.write(str(int(bbox[0]))+' '+ str(int(bbox[1]))+' '+ str(int(bbox[2]))+' '+ str(int(bbox[3]))+' ')
         outputFile.write('\n')
     else:
         outputFile.write(imageFileName +' 0' '\n')
     temp = ftest.readline().strip()
     if temp:
         imageFileName = basePath+'/' + temp
     else:
         break
     vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [1]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = '../models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = '../output/ach.caffemodel'
  

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    basePath = '../data/tfds/ach'
    testFileName = '../data/tfds/ach/Imagelist.txt'
    #demo(net, basePath,testFileName,('target',)) # locate the targets (normal) in images
    demo(net, basePath,testFileName,('fault',)) # locate the faults in images

    plt.show()
