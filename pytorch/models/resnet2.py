from IPython.terminal.debugger import set_trace as keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class BottleNeckA(nn.Module):
    def __init__(self, in_size, ch, out_size, stride=2):
        #w = math.sqrt(2)
        super(BottleNeckA, self).__init__()
        self.conv1=nn.Conv2d(in_size, ch, 1, stride, 0,  bias=False)
        self.bn1=nn.BatchNorm2d(ch)
        self.conv2=nn.Conv2d(ch, ch, 3, 1, 1,  bias=False)
        self.bn2=nn.BatchNorm2d(ch)
        self.conv3=nn.Conv2d(ch, out_size, 1, 1, 0, bias=False)
        self.bn3=nn.BatchNorm2d(out_size)

        self.conv4=nn.Conv2d(in_size, out_size, 1, stride, 0, bias=False)
        self.bn4=nn.BatchNorm2d(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(nn.Module):
    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        self.conv1=nn.Conv2d(in_size, ch, 1, 1, 0, bias=False)
        self.bn1=nn.BatchNorm2d(ch)
        self.conv2=nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2=nn.BatchNorm2d(ch)
        self.conv3=nn.Conv2d(ch, in_size, 1, 1, 0, bias=False)
        self.bn3=nn.BatchNorm2d(in_size)


    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(nn.Module):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [ BottleNeckA(in_size, ch, out_size, stride)]
        for i in range(layer-1):
            links += [BottleNeckB(out_size, ch)]        
        self.links = nn.Sequential(*links)

    def __call__(self, x):
        for l in self.links:
            x = l(x)
        return x


class ResNet34(nn.Module):
    insize = 224
    def __init__(self,classes):
        super(ResNet34, self).__init__()
        self.conv1=nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.res2=Block(layer=3, in_size=64, ch=64, out_size=256, stride=1)
        self.res3=Block(4, 256, 128, 512, 2)
        self.res4=Block(23, 512, 256, 1024, 2)
        self.res5=Block(3, 1024, 512, 2048, 2)
        self.fc=nn.Linear(2048, classes)


    def __call__(self, x):
        out = self.bn1(self.conv1(x))
        out = F.max_pool2d(F.relu(out), 3, stride=2)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = F.avg_pool2d(out, 7, stride=1)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


