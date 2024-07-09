import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv, muti_conv
import numpy as np


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SLRModel(nn.Module):
    def __init__(self, num_classes, c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size=1024, gloss_dict=None, loss_weights=None):
        # num_classes：要分类的类数。
        # c2d_type：2D 卷积模型的类型（例如，“resnet18”）。
        # conv_type：一维卷积模型的类型。
        # use_bn：指示是否使用批量规范化的布尔值。
        # tm_type：时态模型的类型（例如，“BiLSTM”）。
        # hidden_size：时态模型的隐藏大小。
        # gloss_dict：包含光泽信息的字典。
        # loss_weights：指定不同损失权重的字典。
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)  # resnet18
        self.conv2d.fc = Identity()
        self.conv1d = muti_conv.muti_TemporalConv(input_size=512,
                                                  hidden_size=hidden_size,
                                                  conv_type=conv_type,
                                                  use_bn=use_bn,
                                                  num_classes=num_classes)
        self.merge_bn = nn.BatchNorm1d(hidden_size)
        self.merge_Activation = nn.ReLU(inplace=True)
        self.merge_conv1d = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2)
        # self.conv1d = TemporalConv(input_size=512,
        #                            hidden_size=hidden_size,
        #                            conv_type=conv_type,
        #                            use_bn=use_bn,
        #                            num_classes=num_classes)

        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)  # 双向LSTM层
        self.classifier = nn.Linear(hidden_size, self.num_classes)  # 全连接层
        self.classifier2 = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):  # 处理梯度中的异常值，例如 NaN 或无穷大，以确保梯度计算的稳定性
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def transform(self, len):  # 根据帧数制作一个用于初等变换的单位矩阵
        max_dim = 2 * len
        a = np.identity(max_dim)
        x = len
        for i in range(1, max_dim):
            if i % 2 == 1:
                a = np.insert(a, i, values=a[x], axis=0)
                x = x + 2
        b = torch.tensor(a[0:max_dim], dtype=torch.float32)  # 将生成的单位矩阵转换为 PyTorch 的张量

        return b

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos  视频
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features  帧级特征
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)  # 将处理后的帧级特征 framewise 和特征长度列表 len_x 输入到一维卷积模型 conv1d 中进行处理。
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']  # 从一维卷积模型的输出中获取视觉特征 x 和特征长度 lgt。

        tm_outputs = self.temporal_model(x, lgt)  # 将视觉特征 x 和特征长度 lgt 输入到时态模型 temporal_model 中进行处理，得到时态模型的输出。

        outputs = self.classifier(tm_outputs['predictions'])   # 将时态模型的预测结果输入到分类器 classifier 中，得到最终的序列预测结果 outputs。

        # merge = torch.add(x, tm_outputs['predictions'])
        # merge_out = self.classifier2(merge)

        merge = torch.cat((x.transpose(0, 1), tm_outputs['predictions'].transpose(0, 1)), dim=1)
        transform = self.transform(lgt[0]).cuda()
        merge_tr = torch.matmul(transform, merge)
        merge_feat0 = self.merge_conv1d(merge_tr.transpose(1, 2))
        merge_feat1 = self.merge_bn(merge_feat0)
        merge_feat = self.merge_Activation(merge_feat1).transpose(1, 2).transpose(0, 1)
        merge_logits = self.classifier2(merge_feat)

        # 如果是训练阶段，则预测结果置为 None，否则使用解码器 decoder 对分类器输出进行解码，得到预测结果
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        merge_pred = None if self.training \
            else self.decoder.decode(merge_logits, lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "merge_logits": merge_logits,
            "sequence_logits": outputs,  # 最终的序列预测结果，通常由时态模型和分类器组合而成
            "conv_sents": conv_pred,  # 一维卷积模型的预测结果经过解码器解码后得到的句子
            "recognized_sents": pred,   # 最终序列预测结果经过解码器解码后得到的句子
            "merge_sents": merge_pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'MergeCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["merge_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
