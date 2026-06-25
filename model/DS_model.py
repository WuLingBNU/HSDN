# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            init.kaiming_uniform(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            # BatchNorm initialization
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            # LSTM initialization
            for name, param in module.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.constant_(param, 0)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(
                torch.zeros((1, 1, out_channels), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight, gain=1.414)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class gcn(nn.Module):
    def __init__(self, in_dim, out_dim, in_channel):
        super().__init__()
        self.gcn = GraphConvolution(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_channel)

    def forward(self, x, adj):
        if len(adj.shape) < 3:
            adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return self.bn(self.gcn(x, adj))

class DSModel(nn.Module):
    def __init__(self, num_rois, window_size, Nalpha, out_dim: int, num_window: int = 60):
        super(DSModel, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois
        self.Nalpha = Nalpha
        self.out_dim = out_dim
        self.num_window = num_window

        self.hidden_dim = self.window_size
        self.d_model = self.num_rois * self.hidden_dim
        self.ln = nn.LayerNorm(self.d_model)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.window_size, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        )
        self.global_adj = nn.Parameter(torch.randn([num_rois, num_rois], dtype=torch.float32), requires_grad=True)
        self.gcn = gcn(self.hidden_dim, self.hidden_dim, num_rois)
        self.bn = nn.BatchNorm1d(self.num_rois)

        self.fc = nn.Linear(self.hidden_dim, out_dim)

        self.Nbeta = out_dim - self.Nalpha

        self.cross_attention = CrossAttention()

        self.discriminator = Discriminator(input_dim=self.Nalpha + self.Nbeta)

        self.static_feature_extractor = nn.Sequential(nn.Linear(2, 16), nn.BatchNorm1d(self.num_rois), nn.ELU(),
                                                      nn.Linear(16, 1))
        self.fusion_layer = FusionLayer(64)

    def extract_static(self, x):
        assert len(x.shape) == 4  # batch,windows,rois,dim
        s = x.transpose(1, 2)
        mean_static = F.adaptive_avg_pool2d(s, 1).squeeze(-1)
        max_static = F.adaptive_max_pool2d(s, 1).squeeze(-1)
        s = torch.cat((mean_static, max_static), dim=-1)
        s = self.static_feature_extractor(s)
        return s

    def forward(self, x, reverse_alpha=1.0):
        batch, windows, rois, length = x.shape

        encoded_x = (self.mlp(x) + x).reshape(-1, rois, length)

        feature = self.bn(encoded_x)

        adj = self.get_adj(feature)
        h = F.elu(self.gcn(feature, adj)) + feature
        x = self.fc(h).reshape(batch, windows, rois, -1)  # [batch,window,num_rois,length]

        static_x = self.extract_static(x)

        alpha, beta = torch.split(x, [self.Nalpha, self.Nbeta], dim=-1)

        alpha_mask = self.cross_attention(alpha)
        alpha = (alpha_mask.unsqueeze(-1) * alpha).mean(dim=(1, 2))  # m
        sigmd = self.fusion_layer(alpha, static_x.squeeze(-1)).squeeze(-1)

        finalx = sigmd * alpha + (1 - sigmd) * static_x.squeeze(-1)

        beta = beta.mean(dim=(1, 2))

        beta_shuffle = beta[torch.randperm(beta.size(0))]

        return finalx, alpha, beta, alpha_mask, disc_fake, disc_ture
      
    def get_adj(self, x, self_loop=True):
        adj = torch.bmm(x, x.permute(0, 2, 1))
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(x.device)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

class CatPooling(nn.Module):
    def __init__(self):
        super(CatPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2, 16,bias=False), nn.ELU(),
                                nn.Linear(16, 1,bias=False))
        init.kaiming_uniform(self.fc[0].weight)
        init.kaiming_uniform(self.fc[2].weight)


    def forward(self, x):
        # x is [batch,windows,rois,f]
        assert len(x.shape) == 4
        x1 = self.pool1(x)  # [batch,windows,1,1)
        x2 = self.pool2(x)
        x = torch.cat((x1, x2), -2).squeeze(-1)
        x = self.fc(x).squeeze(-1)
        return x


class classifier_linear(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(classifier_linear, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim

        self.classifier = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))

        init.kaiming_uniform(self.classifier[0].weight)
        init.kaiming_uniform(self.classifier[2].weight)

    def forward(self, h):
        out = self.classifier(h)
        return out

