from model.Resnet import ResNet
from IPython.terminal.debugger import set_trace as keyboard
#from __future__ import print_function
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert
import six
from six.moves import queue
from chainer import serializers

import chainer.datasets as datasets
from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob,os
import time

parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
parser.add_argument('--dataset', '-d', default='/home/baxter/dataset/animeface/animeface-character-dataset/thumb/',
                help='The path of dataset')
parser.add_argument('--batchsize', '-b', type=int, default=50,
                help='Number of images in each mini-batch')
#parser.add_argument('--learnrate', '-l', type=float, default=0.05, help='Learning rate for SGD')
parser.add_argument('--alpha1',  type=float, default=0.001, help='Learning rate for Adam')
parser.add_argument('--epoch', '-e', type=int, default=200,
                help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
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

# Set up a neural network to train.
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.



def data_manage_animefacedata(data_path,in_size=224):
    #Data path setup

    folders = sorted(os.listdir(data_path))
    cats = []#Categorys list
    all_data = []
    for folder in folders:
        if os.path.isfile(data_path + folder + "/" + "ignore"):
            #print("Folder "+ folder + "is ignored!")
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
        img = img.transpose(2,0,1)
        img = img[:3, ...]
        #img = np.reshape(img,(3,in_size,in_size))
        imageData.append(img)
        labelData.append(np.int32(label_id))

    threshold = np.int32(len(imageData)/8*7)
    train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
    test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
    return train,test



model = ResNet()

#Check cuda environment and setup models
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(0).use()
    model.to_gpu(0)
    from chainer.cuda import cupy as xp
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



sum_accuracy = 0
sum_loss = 0
epochnum = args.epoch

train_count = len(train)
test_count = len(test)
iteration = 0
printiter = 1
last_time = time.time()
while(train_iter.epoch < epochnum):
    batch = train_iter.next()
    #print len(batch)
    # Reduce learning rate by 0.5 every 25 epochs.
    #if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
    #   optimizer.lr *= 0.5
    #   print('Reducing learning rate to: ', optimizer.lr)

    x_array, t_array = convert.concat_examples(batch, args.gpu)
    x = chainer.Variable(x_array)
    t = chainer.Variable(t_array)

    optimizer.update(model, x, t)



    if iteration%printiter == 0:
            print('iter:{}, loss: {}, accuracy: {}%, time: {}s'.format(
                    iteration,model.loss.data,int(model.accuracy.data*100.),time.time()-last_time))
            with open("./train_log.log","a") as f:
                f.write(str(iteration)+","+str(model.loss.data)+","+str(model.accuracy.data*100))
                f.write("\n")
            print("Average time: "+str((time.time()-last_time)/printiter))
            last_time = time.time()


    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
    #keyboard()

    if train_iter.is_new_epoch:
    #if True:
        print('epoch: ', train_iter.epoch)
        print('train mean loss: {}, accuracy: {}'.format(
            sum_loss / train_count, sum_accuracy / train_count))
        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        #keyboard()
        #model.train = False
        for batch in test_iter:
            x_array, t_array = convert.concat_examples(batch, args.gpu)
            x_test = chainer.Variable(x_array)
            t_test = chainer.Variable(t_array)
            loss_test = model(x_test, t_test)
            sum_loss += float(loss_test.data) * len(t_test.data)
            sum_accuracy += float(model.accuracy.data) * len(t_test.data)

        test_iter.reset()
        #model.train = True
        print('test mean  loss: {}, accuracy: {}'.format(
            sum_loss / test_count, sum_accuracy / test_count))
        sum_accuracy = 0
        sum_loss = 0

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz('mlp.model', model)
        print('save the optimizer')
        serializers.save_npz('mlp.state', optimizer)

    iteration += 1
