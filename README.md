# LR-FTI-FDet

This repository includes the code for the paper "A Unified Light Framework for Real-time Fault Detection of Freight Train Images". The code for LR-FTI-FDet is  based on [Soft-NMS](https://github.com/bharatsingh430/soft-nms) and [Faster RCNN](https://github.com/rbgirshick/py-faster-rcnn).

### Contents

1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

**NOTE** If you are having issues compiling and you are using a recent version of CUDA/cuDNN, please consult [this issue](https://github.com/rbgirshick/py-faster-rcnn/issues/509?_pjax=%23js-repo-pjax-container#issuecomment-284133868) for a workaround

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

For training the end-to-end version of LR-FTI-FDet with the backbone RFDNet, 2G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the LR-FTI-FDet repository

  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/tmpyzhang/LR-FTI-FDet.git
  ```

2. We'll call the directory that you cloned LR-FTI-FDet into ROOT`

3. Build the Cython modules

   ```Shell
   cd $ROOT/lib
   make
   ```

4. Build Caffe and pycaffe

   ```Shell
   cd $ROOT/caffe
   # Now follow the Caffe installation instructions here:
   #   http://caffe.berkeleyvision.org/installation.html
   
   # If you're experienced with Caffe and have all of the requirements installed
   # and your Makefile.config in place, then simply do:
   make -j8 && make pycaffe
   ```

5. Download pre-computed LR-FTI-FDet detectors (`ach` already in the folder `output\`)
   These models were trained on six fault datasets in our paper.

### Demo

To run the demo

```Shell
cd $ROOT
./tools/demo_tfds.py
```

The demo performs detection using a RFDNet network trained for detection on `Angle cock` dataset.

### Beyond the demo: installation for training and testing models

Build your own dataset containing the training, validation, test data. It should have this basic structure

```Shell
$data/                                  # data
$data/VOCdevkit2007/                    # following the format of VOC
$data/VOCdevkit2007/VOC2007/            # image sets, annotations, etc.
```

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the networks described in the paper: RFDNet.  For convenience, we renamed it to "VGG16.v2". (already in the folder `data\imagenet_models`)

### Usage

To train and test a LR-FTI-FDet detector using the approximate joint training method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$ROOT/output`.

```Shell
cd $ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

### Citing
If you find this repository useful in your research, please consider citing:
```
@INPROCEEDINGS{LR-FTI-FDet,
  author={Y. {Zhang} and M. {Liu} and Y. {Yang} and Y. {Guo} and H. {Zhang}},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={ A Unified Light Framework for Real-time Fault Detection of Freight Train Images}, 
  year={2021},
  volume={},
  number={},
  pages={},
  }
```
```
@INPROCEEDINGS{Light-FTI-FDet,
  author={Y. {Zhang} and M. {Liu} and Y. {Chen} and H. {Zhang} and Y. {Guo}},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Real-Time Vision-Based System of Fault Detection for Freight Trains}, 
  year={2020},
  volume={69},
  number={7},
  pages={5274-5284},
  }
```
```
@INPROCEEDINGS{FTI-FDet,
  author={Y. {Zhang} and K. {Lin} and H. {Zhang} and Y. {Guo} and G. {Sun}},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)}, 
  title={A Unified Framework for Fault Detection of Freight Train Images Under Complex Environment}, 
  year={2018},
  volume={},
  number={},
  pages={1348-1352},
  }
```
