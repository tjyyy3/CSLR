import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

class muti_TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(muti_TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.Activation1 = nn.ReLU(inplace=True)
        self.Activation2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=int(2), ceil_mode=False)
        self.pool2 = nn.MaxPool1d(kernel_size=int(2), ceil_mode=False)
        self.Conv_1_1 = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=5, stride=1, padding=0)
        self.Conv_1_2 = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=9, stride=1, padding=0)
        self.Conv_2_1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding=0)
        self.Conv_2_2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=9, stride=1, padding=0)
        self.weight_1_1 = torch.nn.Parameter(torch.ones((2, 1024, 1)))
        self.weight_1_2 = torch.nn.Parameter(torch.ones((2, 1024, 1)))
        self.weight_2_1 = torch.nn.Parameter(torch.ones((2, 1024, 1)))
        self.weight_2_2 = torch.nn.Parameter(torch.ones((2, 1024, 1)))

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K8', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]



        modules = []

        # for layer_idx, ks in enumerate(self.kernel_size):
        #     input_sz = self.input_size if layer_idx == 0 else self.hidden_size
        #     if ks[0] == 'P':
        #         modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
        #     elif ks[0] == 'K':
        #         modules.append(
        #             nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
        #         )
        #         modules.append(nn.BatchNorm1d(self.hidden_size))
        #         modules.append(nn.ReLU(inplace=True))
        # self.temporal_conv = nn.Sequential(*modules)    #一个星将列表解开成两个独立的参数，传入函数，两个星号，是将字典解开成独立的元素作为形参。

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len //= 2
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):

        conv_1_1 = self.Conv_1_1(frame_feat)
        conv_1_2 = self.Conv_1_2(frame_feat)
        pad1 = nn.ReplicationPad2d(padding=(2, 2))
        conv_1_2 = pad1(conv_1_2)
        # conv_1_1d = (self.weight_1_1 * conv_1_1) + (self.weight_1_2 * conv_1_2)
        # conv_1_1d = conv_1_1 + (self.weight_1_2 * conv_1_2)
        conv_1_1d = conv_1_1 + conv_1_2
        conv_1_1d = self.bn1(conv_1_1d)
        conv_1_1d = self.Activation1(conv_1_1d)
        conv_1_1d = self.pool1(conv_1_1d)
        conv_2_1 = self.Conv_2_1(conv_1_1d)
        conv_2_2 = self.Conv_2_2(conv_1_1d)
        pad2 = nn.ReplicationPad2d(padding=(2, 2))
        conv_2_2 = pad2(conv_2_2)
        conv_2_2d = conv_2_1 + conv_2_2
        # conv_2_2d = (self.weight_2_1 * conv_2_1) + (self.weight_2_2 * conv_2_2)
        # conv_2_2d = conv_2_1 + (self.weight_2_2 * conv_2_2)
        conv_2_2d = self.bn2(conv_2_2d)
        conv_2_2d = self.Activation2(conv_2_2d)
        visual_feat = self.pool2(conv_2_2d)


        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
