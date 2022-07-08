import torch
import torch.nn as nn

'''
Append padding to keep concat size & input-output size
'''

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        out = self.block(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.ConvTranspose2d(mid_dim, out_dim, kernel_size=2, stride=2))

    def forward(self, x):
        out = self.block(x)
        return out

class UNet(nn.Module):
    def __init__(self, num_classes, in_dim=3, conv_dim=64, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.conv_dim = conv_dim
        self.criterion = criterion
        self.build_unet()

    def build_unet(self):
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim))
        self.enc2 = DownBlock(self.conv_dim, self.conv_dim*2)
        self.enc3 = DownBlock(self.conv_dim*2, self.conv_dim*4)
        self.enc4 = DownBlock(self.conv_dim*4, self.conv_dim*8)

        self.dec1 = UpBlock(self.conv_dim*8, self.conv_dim*16, self.conv_dim*8)
        self.dec2 = UpBlock(self.conv_dim*16, self.conv_dim*8, self.conv_dim*4)
        self.dec3 = UpBlock(self.conv_dim*8, self.conv_dim*4, self.conv_dim*2)
        self.dec4 = UpBlock(self.conv_dim*4, self.conv_dim*2, self.conv_dim)

        self.last = nn.Sequential(
            nn.Conv2d(self.conv_dim*2, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        enc1 = self.enc1(x) # 16
        enc2 = self.enc2(enc1) # 8
        enc3 = self.enc3(enc2) # 4
        enc4 = self.enc4(enc3) # 2

        center = nn.MaxPool2d(kernel_size=2, stride=2)(enc4)

        dec1 = self.dec1(center) # 4
        dec2 = self.dec2(torch.cat([enc4, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec2], dim=1))
        dec4 = self.dec4(torch.cat([enc2, dec3], dim=1))

        last = self.last(torch.cat([enc1, dec4], dim=1))
        assert x.size(-1) == last.size(-1), 'input size(W)-{} mismatches with output size(W)-{}' \
                                            .format(x.size(-1), output.size(-1))
        assert x.size(-2) == last.size(-2), 'input size(H)-{} mismatches with output size(H)-{}' \
                                            .format(x.size(-1), output.size(-1))
        return last

if __name__ == '__main__':
    sample = torch.randn((2, 3, 32, 32))
    model = UNet(num_classes=2)
    print(model(sample).size())
