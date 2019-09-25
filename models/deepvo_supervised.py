from torch import functional,optim
import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepVO(nn.Module):
    def __init__(self,  num_hidden_lstms, resnet_block=True, num_lstm=2):
        super(DeepVO, self).__init__

        self.hidden_lstms = num_hidden_lstms
        self.num_lstm = num_lstm

        self.conv1 = nn.Conv2d(6, 64, 7,2,padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5,2,padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 3,1)
        self.conv5 = nn.Conv2d(256, 256, 3,1)
        self.conv6 = nn.Conv2d(256, 512, 3,2)
        self.conv7 = nn.Conv2d(512, 512, 3, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 2)
        self.conv10 = nn.Conv2d(512, 512, 3, 1)
        self.conv11 = nn.Conv2d(512, 1024, 3, 2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(1024)


        if resnet_block:
            pass
        else:
            pass

        if num_lstm==1:
            self.rnn1 = nn.LSTMCell(self.conv11.view(-1).shape[1], self.hidden_lstms[0])
            self.h1 = torch.zeros(1, self.hidden_lstms[0])
            self.c1 = torch.zeros(1, self.hidden_lstms[0])
        else:
            self.rnn1 = nn.LSTMCell(self.conv11.view(-1).shape[1], self.hidden_lstms[0])
            self.rnn2 = nn.LSTMCell(self.hidden_lstms[0], self.hidden_lstms[1])
            self.h1 = torch.zeros(1, self.hidden_lstms[0])
            self.c1 = torch.zeros(1, self.hidden_lstms[0])
            self.h2 = torch.zeros(1, self.hidden_lstms[1])
            self.c2 = torch.zeros(1, self.hidden_lstms[1])


    def resnet_block(self, lst=[]):
        pass
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1))
        x = F.relu(self.bn2(self.conv2))
        x = F.relu(self.bn3(self.conv3))
        x = F.relu(self.bn4(self.conv4))
        x = F.relu(self.bn5(self.conv5))
        x = F.relu(self.bn6(self.conv6))
        x = F.relu(self.bn7(self.conv7))
        x = F.relu(self.bn8(self.conv8))
        x = F.relu(self.bn9(self.conv9))
        x = F.relu(self.bn10(self.conv10))
        if self.num_lstm==1:
            x = self.rnn1(x)
        else:
            x = self.rnn1(x)
            x = self.rnn2(x)

        