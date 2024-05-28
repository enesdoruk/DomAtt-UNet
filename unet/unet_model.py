""" Full assembly of the parts to form the complete network """

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet.unet_parts import *
from PIL import Image
from utils.data_loading import BasicDataset
from unet.domatt import Self_Attn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from unet.glob_loc_att import GLAM

    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.att_down3 = Self_Attn(in_dim=512, activation=nn.ReLU())
        self.att_down2 = Self_Attn(in_dim=256, activation=nn.ReLU())
        
        self.glam_x1 = GLAM(in_channels=64, num_reduced_channels=32, feature_map_size=64, kernel_size=5)
        self.glam_x1  = GLAM(in_channels=128, num_reduced_channels=64, feature_map_size=128, kernel_size=5)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x4_att, _ = self.att_down3(x4)
        x3_att, _ = self.att_down2(x3)
        
        x = self.up1(x5, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return (logits, x)


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input = '/AI/OZ-UNet/data/imgs/ave-0000-0001.jpg'
    img = Image.open(input)
    img = torch.from_numpy(BasicDataset.preprocess(None, img, 0.2, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    net = UNet(n_channels=3, n_classes=86, bilinear=False)
    net.to(device=device)
    
    output, feat = net(img)
    
    feat_ssim = feat.mean(dim=1).unsqueeze(dim=0)
    img_ssim = img.mean(dim=1).unsqueeze(dim=0)
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)