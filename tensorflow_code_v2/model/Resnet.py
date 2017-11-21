import tensorflow as tf
from .links import *
from .BaseModel import BaseModel
from .links.Convolution2D import Convolution2D
from .links.BatchNormalization import BatchNormalization
from .links.Linear import Linear
import math

class BottleNeckA:
    def __init__(self, name,in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        with tf.variable_scope(name):
            self.conv1=Convolution2D('conv1',in_size, ch, 1, stride=stride, pad='VALID', wscale=w, nobias=False)
            #self.bn1=BatchNormalization('bn1',ch,dim=3),
            self.conv2=Convolution2D('conv2',ch, ch, 3, 1, pad='SAME', wscale=w, nobias=True)
            #self.bn2=BatchNormalization('bn2',ch,dim=3),
            self.conv3=Convolution2D('conv3',ch, out_size, 1, 1, pad='VALID', wscale=w, nobias=True)
            #self.bn3=BatchNormalization('bn3',out_size,dim=3),
            self.conv4=Convolution2D('conv4',in_size, out_size, 1, stride, pad='VALID', wscale=w, nobias=True)
            #self.bn4=BatchNormalization('bn4',out_size,dim=3)

    def __call__(self, x):
        #h1 = tf.nn.relu(self.bn1(self.conv1(x)))
        h1 = tf.nn.relu(tf.layers.batch_normalization(self.conv1(x)))
        #h1 = tf.nn.relu(self.bn2(self.conv2(h1)))
        h1 = tf.nn.relu(tf.layers.batch_normalization(self.conv2(h1)))
        #h1 = self.bn3(self.conv3(h1))
        h1 = tf.layers.batch_normalization(self.conv3(h1))
        #h2 = self.bn4(self.conv4(x))
        h2 = tf.layers.batch_normalization(self.conv4(x))
        return tf.nn.relu(h1 + h2)

class BottleNeckB:
    def __init__(self,name, in_size, ch):
        w = math.sqrt(2)
        with tf.variable_scope(name):
            self.conv1=Convolution2D('conv1',in_size, ch, 1, 1, pad='VALID', wscale=w, nobias=True)
            self.conv2=Convolution2D('conv2',ch, ch, 3, 1,pad='SAME', wscale=w, nobias=True)
            self.conv3=Convolution2D('conv3',ch, in_size, 1, 1, pad='VALID', wscale=w, nobias=True)


    def __call__(self, x):
        #h = tf.nn.relu(self.bn1(self.conv1(x)))
        h = tf.nn.relu(tf.layers.batch_normalization(self.conv1(x)))
        #h = tf.nn.relu(self.bn2(self.conv2(h)))
        h = tf.nn.relu(tf.layers.batch_normalization(self.conv2(h)))
        #h = self.bn3(self.conv3(h))
        h = tf.layers.batch_normalization(self.conv3(h))
        return tf.nn.relu(h + x)

class Block:
    def __init__(self, name,layer, in_size, ch, out_size, stride=2):
        with tf.variable_scope(name):
            links = [BottleNeckA('a',in_size, ch, out_size, stride)]
            for i in range(layer-1):
                links += [BottleNeckB('b{}'.format(i+1),out_size, ch)]
            #print str(links)
            self.links = links
    def forward(self, x):
        #print x
        for l in self.links:
            x = l(x)
        return x



class ResNet(BaseModel):
    def __init__(self, name, in_channels,out_classes,seed=1):
        super(ResNet, self).__init__(name,seed)
        with tf.variable_scope(self.name):
            self.conv1 = Convolution2D('conv1',in_channels,64,7,stride=2,pad='VALID',nobias=False)
            #self.bn1 = tf.layers.batch_normalization(64)
            self.res2=Block('res2',layer=3, in_size=64, ch=64, out_size=256, stride=2)
            self.res3=Block('res3',4, 256, 128, 512, 2)
            self.res4=Block('res4',23, 512, 256, 1024, 2)
            self.res5=Block('res5',3, 1024, 512, 2048, 2)
            self.fc=Linear('linear',2048, out_classes)
        self.train = True


    def __call__(self, x):
        #h = self.bn1(self.conv1(x))
        h = tf.layers.batch_normalization(self.conv1(x))
        #h = tf.nn.max_pool(tf.nn.relu(h), ksize=[1,3,3,1], strides=2,padding='SAME')
        h = tf.layers.max_pooling2d(tf.nn.relu(h),pool_size=[3,3],strides=[2,2],padding='same')
        h = self.res2.forward(h)
        h = self.res3.forward(h)
        h = self.res4.forward(h)
        h = self.res5.forward(h)
        #h = tf.nn.avg_pool(h, ksize=[1,7,7,1], strides=1,padding='VALID')
        h = tf.layers.average_pooling2d(tf.nn.relu(h),pool_size=[3,3],strides=[2,2],padding='valid')
        h = tf.contrib.layers.flatten(h)
        h = self.fc(h)
        return h

