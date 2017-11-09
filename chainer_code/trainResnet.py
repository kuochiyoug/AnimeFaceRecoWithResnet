from model.Resnet import ResNet

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

from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob,os


def data_manage_animefacedata(data_path="/home/koma/dataset/animeface-character-dataset/thumb/",in_size=224):
    #Data path setup
   
    #data_path = "/home/koma/dataset/animeface-character-dataset/thumb/"
    #test_data_path = "/data/test/"
    
    #folders = glob.glob(data_path+"*")
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
cuda.check_cuda_available()

model = ResNet()
cuda.get_device(0).use()
model.to_gpu(0)
print("Model Ready with GPU")

#Setup optimizer
optimizer = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
print("Optimizer Sets!")

#Setup train data and test data
train, test = data_manage_animefacedata()
train_iter = chainer.iterators.SerialIterator(train,20)
test_iter = chainer.iterators.SerialIterator(test,10,repeat=False,shuffle=False)
print("Data Loaded!")


# In[ ]:


sum_accuracy = 0
sum_loss = 0
epochnum = 10

train_count = len(train)
test_count = len(test)
#while(train_iter.epoch < args.epoch):
while(train_iter.epoch < epochnum):
    batch = train_iter.next()
    #print len(batch)
    print('train mean loss: {}, accuracy: {}'.format(
            sum_loss / train_count, sum_accuracy / train_count))
    # Reduce learning rate by 0.5 every 25 epochs.
    #if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
    #   optimizer.lr *= 0.5
    #   print('Reducing learning rate to: ', optimizer.lr)

    #x_array, t_array = convert.concat_examples(batch, args.gpu)
    x_array, t_array = convert.concat_examples(batch, 0)
    x = chainer.Variable(x_array)
    t = chainer.Variable(t_array)
    optimizer.update(model, x, t)
    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
    
    if train_iter.is_new_epoch:
        print('epoch: ', train_iter.epoch)
        print('train mean loss: {}, accuracy: {}'.format(
            sum_loss / train_count, sum_accuracy / train_count))
        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        #model.predictor.train = False
        for batch in test_iter:
            x_array, t_array = convert.concat_examples(batch, 0)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            loss = model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        test_iter.reset()
        #model.predictor.train = True
        print('test mean  loss: {}, accuracy: {}'.format(
            sum_loss / test_count, sum_accuracy / test_count))
        sum_accuracy = 0
        sum_loss = 0

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz('mlp.model', model)
        print('save the optimizer')
        serializers.save_npz('mlp.state', optimizer)

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