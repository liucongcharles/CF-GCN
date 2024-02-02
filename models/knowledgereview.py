import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(self, channel):
        super(PredictionHead, self).__init__()
        self.mask_generation = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

    def forward(self, feature):
        mask = self.mask_generation(feature)

        return mask


class FeatureEnhancementUnit(nn.Module):
    def __init__(self, in_channel, channel):
        super(FeatureEnhancementUnit, self).__init__()
        self.feature_transition = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        hid_channel = max(8, channel // 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channel, hid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channel, in_channel),
            nn.Sigmoid()
        )
        self.feature_context = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        x = self.feature_transition(x)
        if mask is not None:
            x = x * mask + x
        B, C, _, _ = x.size()
        vec_y = self.avg_pool(x).view(B, C)
        channel_att = self.channel_attention(vec_y).view(B, C, 1, 1)
        feu_out = self.feature_context(x * channel_att)

        return feu_out


class KnowledgeReviewModule(nn.Module):
    def __init__(self, in_channel, channel):
        super(KnowledgeReviewModule, self).__init__()
        #
        self.feu_1 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_2 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_3 = FeatureEnhancementUnit(in_channel, channel)
        #
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel + 1, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, fine_mask, coarse_mask):
        # without attention
        context_1 = self.feu_1(feature)
        # reverse attention
        reverse_att = 1 - torch.sigmoid(coarse_mask)
        context_2 = self.feu_2(feature, reverse_att)
        # uncertainty attention
        uncertainty_att = (1 - torch.sigmoid(coarse_mask)) * torch.sigmoid(fine_mask) + \
                          (1 - torch.sigmoid(fine_mask)) * torch.sigmoid(coarse_mask)
        context_3 = self.feu_3(feature, uncertainty_att)
        feature = context_1 + context_2 + context_3
        krm_out = self.fusion_conv(torch.cat([feature, fine_mask], dim=1))

        return krm_out


