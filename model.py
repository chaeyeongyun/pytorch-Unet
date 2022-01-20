import torch
import torch.nn as nn


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

class ContractingPath(nn.Module):
    def __init__(self, in_channels, first_outchannels):
        super(ContractingPath, self).__init__()
        self.conv1 = DBConv(in_channels, first_outchannels)
        self.conv2 = DBConv(first_outchannels, first_outchannels*2)
        self.conv3 = DBConv(first_outchannels*2, first_outchannels*4)
        self.conv4 = DBConv(first_outchannels*4, first_outchannels*8)

        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        output1 = self.conv1(x) # (N, 64, 568, 568)
        output = self.maxpool(output1) # (N, 64, 284, 284)
        output2 = self.conv2(output) # (N, 128, 280, 280)
        output = self.maxpool(output2) # (N, 128, 140, 140)
        output3 = self.conv3(output) # (N, 256, 136, 136)
        output = self.maxpool(output3) # (N, 256, 68, 68)
        output4 = self.conv4(output) # (N, 512, 64, 64)
        output = self.maxpool(output4) # (N, 512, 32, 32)
        return output1, output2, output3, output4, output


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
        
        self.upconv3 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2) # (N, 128, 200, 200)
        self.conv3 = DBConv(in_channels//4, in_channels//8) # (N, 128, 196, 196)

        self.upconv4 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2) # (N, 64, 392, 392)
        self.conv4 = DBConv(in_channels//8, in_channels//16) # (N, 64, 388, 388)

    def forward(self, x, pass1, pass2, pass3, pass4):
        output = self.upconv1(x)
        diffY = pass4.size()[2] - output.size()[2]
        diffX = pass4.size()[3] - output.size()[3]
        pass4 = nn.ConstantPad2d((diffX//2, diffX//2, diffY//2, diffY//2))
        output = torch.cat((output, pass4), 1)
        output = self.conv1(output)
        
        output = self.upconv2(output)
        diffY = pass3.size()[2] - output.size()[2]
        diffX = pass3.size()[3] - output.size()[3]
        pass3 = nn.ConstantPad2d((diffX//2, diffX//2, diffY//2, diffY//2))
        output = torch.cat((output, pass3), 1)
        output = self.conv2(output)
        
        output = self.upconv3(output)
        diffY = pass2.size()[2] - output.size()[2]
        diffX = pass2.size()[3] - output.size()[3]
        pass2 = nn.ConstantPad2d((diffX//2, diffX//2, diffY//2, diffY//2))
        output = torch.cat((output, pass2), 1)
        output = self.conv3(output)
        
        output = self.upconv4(output)
        diffY = pass1.size()[2] - output.size()[2]
        diffX = pass1.size()[3] - output.size()[3]
        pass1 = nn.ConstantPad2d((diffX//2, diffX//2, diffY//2, diffY//2))
        output = torch.cat((output, pass1), 1)
        output = self.conv4(output)

        return output

class Unet(nn.Module):
    def __init__(self, in_channels=3, first_outchannels=64, num_classes=2):
        super(Unet, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels, first_outchannels=first_outchannels)
        self.middle_conv = DBConv(first_outchannels*8, first_outchannels*16)
        self.expansive_path = ExpansivePath(in_channels=first_outchannels*16)
        self.conv_1x1 = nn.Conv2d(first_outchannels//16, num_classes, 1)
    
    def forward(self, x):
        pass1, pass2, pass3, pass4, output = self.contracting_path(x)
        output = self.middle_conv(output)
        output = self.expansive_path(output, pass1, pass2, pass3, pass4)
        output = self.conv_1x1(output)

        return output


if __name__ == '__main__':
    unet = Unet()