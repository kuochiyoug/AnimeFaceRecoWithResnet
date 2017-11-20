from IPython.terminal.debugger import set_trace as keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
"""

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



"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
"""

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

        self.links = links

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




"""
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        print("num_classes = " + str(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        keyboard()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
"""

