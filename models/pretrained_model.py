import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DBConv(nn.Module):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''
    def __init__(self, in_channels, out_channels):
        super(DBConv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)


class ExpansivePath(nn.Module):
    '''
    pass1, pass2, pass3, pass4 are the featuremaps passed from Contracting path
    '''
    def __init__(self, in_channels):
        super(ExpansivePath, self).__init__()
        # input (N, 1024, 28, 28)
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2) # (N, 512, 56, 56)
        self.conv1 = DBConv(in_channels, in_channels//2) # (N, 512, 52, 52)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels//2, in_channels//4, 2, 2) # (N, 256, 104, 104)
        self.conv2 = DBConv(in_channels//2, in_channels//4) # (N, 256, 100, 100)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels//4, in_channels//8, 2, 2) # (N, 128, 200, 200)
        self.conv3 = DBConv(in_channels//4, in_channels//8) # (N, 128, 196, 196)

        self.upconv4 = nn.ConvTranspose2d(in_channels//8, in_channels//16, 2, 2) # (N, 64, 392, 392)
        self.conv4 = DBConv(in_channels//8, in_channels//16) # (N, 64, 388, 388)
        
        # for match output shape with 

    def forward(self, x, pass1, pass2, pass3, pass4):
        # input (N, 1024, 28, 28)
        output = self.upconv1(x)# (N, 512, 56, 56)
        diffY = pass4.size()[2] - output.size()[2]
        diffX = pass4.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 512, 64, 64)
        output = torch.cat((output, pass4), 1) # (N, 1024, 64, 64)
        output = self.conv1(output) # (N, 512, 60, 60)
        
        output = self.upconv2(output) # (N, 256, 120, 120)
        diffY = pass3.size()[2] - output.size()[2]
        diffX = pass3.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 256, 136, 136)
        output = torch.cat((output, pass3), 1) # (N, 512, 136, 136)
        output = self.conv2(output) # (N, 256, 132, 132)
        
        output = self.upconv3(output) # (N, 128, 264, 264)
        diffY = pass2.size()[2] - output.size()[2]
        diffX = pass2.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 128, 280, 280)
        output = torch.cat((output, pass2), 1) # (N, 256, 280, 280)
        output = self.conv3(output) # (N, 128, 276, 276)
        
        output = self.upconv4(output) # (N, 64, 552, 552)
        diffY = pass1.size()[2] - output.size()[2]
        diffX = pass1.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 64, 568, 568)
        output = torch.cat((output, pass1), 1) # (N, 128, 568, 568)
        output = self.conv4(output) # (N, 64, 564, 564)
        
        return output

class ResNetUnet(nn.Module):
    def __init__(self, in_channels=3, first_outchannels=64, num_classes=3, init_weights=True):
        super(ResNetUnet, self).__init__()
        # self.contracting_path = ContractingPath(in_channels=in_channels, first_outchannels=first_outchannels)
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = self.convrelu(64, 64, 1, 0) # (N, 64, x.H/2, x.W/2)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = self.convrelu(64, 64, 1, 0)       
        
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = self.convrelu(128, 128, 1, 0)  
        
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = self.convrelu(256, 256, 1, 0)  
        
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = self.convrelu(512, 512, 1, 0)  
        
        self.middle_conv = DBConv(first_outchannels*8, first_outchannels*16)
        self.expansive_path = ExpansivePath(in_channels=first_outchannels*16)
        self.conv_1x1 = nn.Conv2d(first_outchannels, num_classes, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if init_weights:
            print('initialize weights...')
            self._initialize_weights()
    
    def convrelu(self, in_channels, out_channels, kernel, padding):
        layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                    nn.ReLU(inplace=True),
                )
        return layer
        
    def forward(self, x):
        (input_h, input_w) = x.shape[2], x.shape[3]
        layer0 = self.layer0(x)
        
        pass1 = self.upsample(layer0)
        layer0 = self.layer0_1x1(layer0)            
        
        layer1 = self.layer1(layer0)
        layer1 = self.layer1_1x1(layer1) 
        
        layer2 = self.layer2(layer1)
        pass2 = self.upsample(layer2)
        layer2 = self.layer2_1x1(layer2) 
        
        layer3 = self.layer3(layer2)        
        pass3 = self.upsample(layer3)
        layer3 = self.layer3_1x1(layer3) 
        
        layer4 = self.layer4(layer3)
        pass4 = self.upsample(layer4)
        layer4 = self.layer4_1x1(layer4)
        
        output1 = self.middle_conv(layer4)
        output2 = self.expansive_path(output1, pass1, pass2, pass3, pass4)
        output3 = self.conv_1x1(output2)

        return output3
   
    def _initialize_weights(self):
        for m in self.middle_conv.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.expansive_path.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
            


if __name__ == '__main__':
    input = torch.zeros((2, 3, 520, 520))
    unet = ResNetUnet()
    unet(input)