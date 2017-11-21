from __future__ import print_function
from IPython.terminal.debugger import set_trace as keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from models.resnet2 import ResNet34

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Cuda is ready")
else:
    print("No usable Cuda, use CPU")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(224, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.Scale((224,224)),
    #transforms.Resize((32,32)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
batch_size=50
trainset = torchvision.datasets.ImageFolder(root='/home/koma/dataset/animeface/animeface-character-dataset/thumb',transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

class_num = len(trainset.classes)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = ResNet34(class_num)



if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #net = torch.nn.DataParallel(net, device_ids=[3])
    cudnn.benchmark = True



criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(),lr=args.lr)
print("Optimizer Setup Finish")


epoch_num =200
# Training

#keyboard()
iteration = 0
for epoch in range(epoch_num):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            #keyboard()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        #train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        #total += targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        if iteration%10 == 0:
            print (str(iteration)+" "+ str(loss.data[0])+" ",str(float(correct)/batch_size*100))
            #imgplot = plt.imshow(final_image[0])
            with open("./train_log.log","a") as f:
                f.write(str(iteration)+","+str(loss.data[0])+","+str(float(correct)/batch_size*100))
                f.write("\n")
        iteration += 1
        if iteration == 10000:
            break
    if iteration == 10000:
        break


    #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



