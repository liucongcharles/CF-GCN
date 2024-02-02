import torch
import torch.nn.functional as F
import torch.nn as nn


BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v_y = nn.Conv2d(plane*2, plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q_y = nn.Conv2d(plane*2, plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)

        self.conv_wg_decode = nn.Conv1d(inter_plane*2, inter_plane*2, kernel_size=1, bias=False)
        self.bn_wg_decode = BatchNorm1d(inter_plane*2)

        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.outdecode = nn.Sequential(nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.xpre = nn.Sequential(nn.Conv2d(inter_plane*4, inter_plane*2, kernel_size=1),
                                 BatchNorm2d(inter_plane*2),
                                 nn.Conv2d(inter_plane*2, plane, kernel_size=1),
                                 BatchNorm2d(plane)
                                  )

    def forward(self, x, y):
        if y is None:
            node_k = self.node_k(x)
            node_v = self.node_v(x)
            node_q = self.node_q(x)
        else:
            node_k = y
            node_v = self.node_v_y(x)
            node_q = self.node_q_y(x)
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        if y is None:
            AVW = self.conv_wg(AV)
            AVW = self.bn_wg(AVW)
        else:
            AVW = self.conv_wg_decode(AV)
            AVW = self.bn_wg_decode(AVW)
        AVW = AVW.view(b, c, h, -1)
        if y is None:
            out = F.relu_(self.out(AVW) + x)
        else:
            out = F.relu_(self.outdecode(AVW) + self.xpre(x))
        return out


class CFGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """

    def __init__(self, planes, ratio=4):
        super(CFGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        # decode theta
        self.thetadecode = nn.Conv2d(planes//2, planes // ratio , kernel_size=1, bias=False)
        self.bn_thetadecode = BatchNorm2d(planes // ratio )

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.localdecode = nn.Sequential(
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2),
            nn.Conv2d(planes//2, planes//2, 3, groups=planes//2, stride=2, padding=1, bias=False),
            BatchNorm2d(planes//2))
        self.gcn_local_attention = SpatialGCN(planes)
        self.gcn_local_attentiondecode = SpatialGCN(planes//2)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))
        self.finaldecode = nn.Sequential(nn.Conv2d(192, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat, featdecode):
        # # # # Local # # # #
        x = feat
        if featdecode is None:
            local = self.local(feat)
            local = self.gcn_local_attention(local, None)
        else:
            localtoken = self.localdecode(featdecode)
            localfeat = self.local(feat)
            local = self.gcn_local_attentiondecode(localfeat, localtoken)
        if featdecode is None:
            local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
            spatial_local_feat = x * local + x
        else:
            local = F.interpolate(local, size=featdecode.size()[2:], mode='bilinear', align_corners=True)
            spatial_local_feat = featdecode * local + featdecode

        # # # # Projection Space # # # #
        x_sqz, b = x, x
        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        if featdecode is None:
            # decoder
            b = self.theta(b)
            b = self.bn_theta(b)
            b = self.to_matrix(b)
        else:
            # encoder
            b = self.thetadecode(featdecode)
            b = self.bn_thetadecode(b)
            b = self.to_matrix(b)
        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        z = z_idt + z

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x + y)

        # cat or sum, nearly the same results
        if featdecode is None:
            out = self.final(torch.cat((spatial_local_feat, g_out), 1))
        else:
            out = self.finaldecode(torch.cat((spatial_local_feat, g_out), 1))

        return out


class CFGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(CFGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = CFGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes//2, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes//2),
                                   nn.ReLU(interplanes//2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )


    def forward(self, x, y):
        if y is None:
            # gcnencoder
            output = self.conva(x)
            output = self.dualgcn(output, None)
        else:
            # gcndecoder
            output = self.conva(x)
            output = self.dualgcn(output, y)
        output = self.convb(output)
        return output

