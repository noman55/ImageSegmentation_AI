
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
