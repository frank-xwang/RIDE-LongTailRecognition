"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, num_classes=1000, use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, top_choices_num=5, pos_weight=20, share_expert_help_pred_fc=True, force_all=False, s=30):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            s = 1
            self.linears = nn.ModuleList([nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])

        self.num_classes = num_classes

        self.top_choices_num = top_choices_num
        
        self.share_expert_help_pred_fc = share_expert_help_pred_fc
        self.layer4_feat = True

        expert_hidden_fc_output_dim = 16
        self.expert_help_pred_hidden_fcs = nn.ModuleList([nn.Linear((layer4_output_dim if self.layer4_feat else layer3_output_dim) * block.expansion, expert_hidden_fc_output_dim) for _ in range(self.num_experts - 1)])
        if self.share_expert_help_pred_fc:
            self.expert_help_pred_fc = nn.Linear(expert_hidden_fc_output_dim + self.top_choices_num, 1)
        else:
            self.expert_help_pred_fcs = nn.ModuleList([nn.Linear(expert_hidden_fc_output_dim + self.top_choices_num, 1) for _ in range(self.num_experts - 1)])

        self.pos_weight = pos_weight

        self.s = s

        self.force_all = force_all # For calulating FLOPs

        if not force_all:
            for name, param in self.named_parameters():
                if "expert_help_pred" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)
        if not self.layer4_feat:
            self.feat = x
        x = (self.layer4s[ind])(x)
        if self.layer4_feat:
            self.feat = x

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)

        if self.use_dropout:
            x = self.dropout(x)

        x = (self.linears[ind])(x)
        x = x * self.s # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
        return x

    def pred_expert_help(self, input_part, i):
        feature, logits = input_part
        feature = F.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)
        feature = feature / feature.norm(dim=1, keepdim=True)

        feature = self.relu((self.expert_help_pred_hidden_fcs[i])(feature))

        topk, _ = torch.topk(logits, k=self.top_choices_num, dim=1)
        confidence_input = torch.cat((topk, feature), dim=1)
        if self.share_expert_help_pred_fc:
            expert_help_pred = self.expert_help_pred_fc(confidence_input)
        else:
            expert_help_pred = (self.expert_help_pred_fcs[i])(confidence_input)
        return expert_help_pred

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        shared_part = self.layer2(x)
        if self.share_layer3:
            shared_part = self.layer3(shared_part)
        
        if target is not None: # training time
            output = shared_part.new_zeros((shared_part.size(0), self.num_classes))

            expert_help_preds = output.new_zeros((output.size(0), self.num_experts - 1), dtype=torch.float) 
            # first column: correctness of the first model, second: correctness of expert of the first and second, etc.
            correctness = output.new_zeros((output.size(0), self.num_experts), dtype=torch.uint8)

            loss = output.new_zeros((1,))
            for i in range(self.num_experts):
                output += self._separate_part(shared_part, i)
                correctness[:, i] = output.argmax(dim=1) == target # Or: just helpful, predict 1
                if i != self.num_experts - 1:
                    expert_help_preds[:, i] = self.pred_expert_help((self.feat, output / (i+1)), i).view((-1,))

            for i in range(self.num_experts - 1):
                # import ipdb; ipdb.set_trace()
                expert_help_target = (~correctness[:, i]) & correctness[:, i+1:].any(dim=1)
                expert_help_pred = expert_help_preds[:, i]
                
                print("Helps ({}):".format(i+1), expert_help_target.sum().item() / expert_help_target.size(0))
                print("Prediction ({}):".format(i+1), (torch.sigmoid(expert_help_pred) > 0.5).sum().item() / expert_help_target.size(0), (torch.sigmoid(expert_help_pred) > 0.3).sum().item() / expert_help_target.size(0))
                
                loss += F.binary_cross_entropy_with_logits(expert_help_pred, expert_help_target.float(), pos_weight=expert_help_pred.new_tensor([self.pos_weight]))
            
            # output with all experts
            return output / self.num_experts, loss / (self.num_experts - 1)
        else: # test time
            expert_next = shared_part.new_ones((shared_part.size(0),), dtype=torch.uint8)
            num_experts_for_each_sample = shared_part.new_ones((shared_part.size(0), 1), dtype=torch.long)
            output = self._separate_part(shared_part, 0)
            for i in range(1, self.num_experts):
                expert_help_pred = self.pred_expert_help((self.feat, output[expert_next] / i), i-1).view((-1,))
                if not self.force_all: # For evaluating FLOPs
                    expert_next[expert_next.clone()] = (torch.sigmoid(expert_help_pred) > 0.5).type(torch.uint8)
                print("expert ({}):".format(i), expert_next.sum().item() / expert_next.size(0))
                
                if not expert_next.any():
                    break
                output[expert_next] += self._separate_part(shared_part[expert_next], i)
                num_experts_for_each_sample[expert_next] += 1
            
            return output / num_experts_for_each_sample.float(), num_experts_for_each_sample

        return output
