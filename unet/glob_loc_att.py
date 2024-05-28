import torch
from torch import nn

class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)
        
        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att
    

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, 1, 1)
        
    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1)
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)
        
        query_key = torch.bmm(key.permute(0, 2, 1), query)
        query_key = query_key.reshape(N, -1).softmax(-1)
        query_key = query_key.reshape(N, int(H*W), int(H*W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        att = torch.bmm(value, query_key).reshape(N, C, H, W)
        att = self.conv1x1_att(att)
        
        return (global_channel_output * att) + global_channel_output
    

class LocalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)
    
    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x
    
class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels*4), 1, 1, 1)
        
        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=3, dilation=3)
        
    def forward(self, feature_maps, local_channel_output):
        att = self.conv1x1_1(feature_maps)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.conv1x1_2(att)
        return (local_channel_output * att) + local_channel_output
    
    
class GLAM(nn.Module):
    
    def __init__(self, in_channels, num_reduced_channels, feature_map_size, kernel_size):
        '''
        Song, C. H., Han, H. J., & Avrithis, Y. (2022). All the attention you need: Global-local, spatial-channel attention for image retrieval. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2754-2763).

        Args:
            in_channels (int): number of channels of the input feature map
            num_reduced_channels (int): number of channels that the local and global spatial attention modules will reduce the input feature map. Refer to figures 3 and 5 in the paper.
            feaure_map_size (int): height/width of the feature map
            kernel_size (int): scope of the inter-channel attention
        '''
        
        super().__init__()
        
        self.local_channel_att = LocalChannelAttention(feature_map_size, kernel_size)
        self.local_spatial_att = LocalSpatialAttention(in_channels, num_reduced_channels)
        self.global_channel_att = GlobalChannelAttention(feature_map_size, kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels, num_reduced_channels)
        
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333, 0.333, 0.333])) # equal intial weights
        
    def forward(self, x):
        local_channel_att = self.local_channel_att(x) # local channel
        local_att = self.local_spatial_att(x, local_channel_att) # local spatial
        global_channel_att = self.global_channel_att(x) # global channel
        global_att = self.global_spatial_att(x, global_channel_att) # global spatial
        
        local_att = local_att.unsqueeze(1) # unsqueeze to prepare for concat
        global_att = global_att.unsqueeze(1) # unsqueeze to prepare for concat
        x = x.unsqueeze(1) # unsqueeze to prepare for concat
        
        all_feature_maps = torch.cat((local_att, x, global_att), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(1)
        
        return fused_feature_maps