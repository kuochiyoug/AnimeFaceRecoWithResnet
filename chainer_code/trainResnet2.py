from model.Resnet import ResNet

#from __future__ import print_function
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.cuda import cupy as xp
from chainer.dataset import convert
import six
from six.moves import queue
from chainer import serializers
from chainer.training import triggers
import chainer.datasets as datasets
from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob,os

parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
parser.add_argument('--dataset', '-d', default='/home/koma/dataset/animeface-character-dataset/thumb/',
                help='The path of dataset')
parser.add_argument('--batchsize', '-b', type=int, default=64,
                help='Number of images in each mini-batch')
#parser.add_argument('--learnrate', '-l', type=float, default=0.05, help='Learning rate for SGD')
parser.add_argument('--alpha1',  type=float, default=0.005, help='Learning rate for SGD')
parser.add_argument('--epoch', '-e', type=int, default=300,
                help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=0,
                help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                help='Resume the training from snapshot')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')


def data_manage_animefacedata(data_path,in_size=224):
    #Data path setup
    folders = sorted(os.listdir(data_path))
    cats = []#Categorys list
    all_data = []
    for folder in folders:
        if os.path.isfile(data_path + folder + "/" + "ignore"):
            print("Folder "+ folder + "is ignored!")
            continue
        else:
            cats.append(folder)
            label = folder
            img_filelist = glob.glob(data_path + folder + "/"+"*.png")
            for imgfile in img_filelist:
                all_data.append([imgfile,label])
    print("labels="+str(len(cats)))

    all_data = np.random.permutation(all_data) #Random the rank

    imageData = []
    labelData = []
    for PathAndLabel in all_data:
        img = Image.open(PathAndLabel[0])
        img = img.resize((in_size,in_size))
        label_id = cats.index(PathAndLabel[1])
        #print PathAndLabel[1]
        img = np.asarray(np.float32(img))
        if img.shape[2] != 3:
            continue
        img = np.reshape(img,(3,in_size,in_size))
        imageData.append(img)
        labelData.append(np.int32(label_id))

    threshold = np.int32(len(imageData)/8*7)
    train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
    test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
    return train,test



#Check cuda environment and setup models

model = ResNet()

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(0).use()
    model.to_gpu(0)
    print("Model Ready with GPU")
else:
    print("Model Ready with CPU")

#Setup optimizer
optimizer = optimizers.Adam(alpha=args.alpha1, beta1=0.5)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
print("Optimizer Sets!")

#Setup train data and test data
train, test = data_manage_animefacedata(args.dataset)
train_iter = chainer.iterators.SerialIterator(train,20)
test_iter = chainer.iterators.SerialIterator(test,10,repeat=False,shuffle=False)
print("Data Loaded!")



updater = training.StandardUpdater(
    train_iter, optimizer, device=args.gpu)

trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='AnimeFace-result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.observe_lr())

trainer.extend(extensions.PrintReport(
    ['epoch',
     'main/loss',
     'main/accuracy',
     'validation/main/loss',
     'validation/main/accuracy',
     'elapsed_time',
     'lr']))
trainer.extend(extensions.PlotReport(
        ['main/loss',
         'validation/main/loss'],
        'epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
        ['main/accuracy',
         'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))

trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
"""
def lr_drop(trainer):
    trainer.updater.get_optimizer('main').lr *= lr_drop_ratio
trainer.extend(
    lr_drop,
    trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))
"""
trainer.run()

"""
updater = training.StandardUpdater(train_iter, optimizer,device=0)
trainer = training.Trainer(updater, (101, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
#trainer.extend(extensions.dump_graph(root_name=))
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
"""
