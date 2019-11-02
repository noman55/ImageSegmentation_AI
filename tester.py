"""1. Getting Started with FCN Pre-trained Models
==============================================

This is a quick demo of using GluonCV FCN model on PASCAL VOC dataset.
Please follow the `installation guide <../index.html>`_ to install MXNet and GluonCV if not yet.
"""
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
import cv2
# using cpu
ctx = mx.cpu(0)

filename= '71.jpeg'
# load the image
img = image.imread(filename)

from matplotlib import pyplot as plt
#plt.imshow(img.asnumpy())
#plt.show()
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)
# get pre-trained model
model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'pascal_voc')
mask.save('output.png')



##############################################################################
# More Examples
# -------------
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/4.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/4.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/5.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/5.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/6.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/6.png
#    :width: 45%
