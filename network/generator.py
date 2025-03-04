import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pdb 


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels,negative_slope):
        super(DownSample,self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels, out_channels,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample,self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self,in_channels = 25, out_channels = 4, features = [64,128,256,512,512]): 
        # in: 22 + 3, clothing-agonistic personal representation p + target clothes image c 
        # out: 3 + 1, synthesize image I' & segmentation mask M
        super(Generator, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottle_neck = nn.Sequential(
            DownSample(in_channels=512, out_channels=512, negative_slope = 0.2),
            UpSample(in_channels=512,out_channels=512),
            nn.Dropout(p=0.5)
        )
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=features[0] * 2 ,out_channels=out_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )


        # Encoder
        for idx,feature in enumerate(features):
            if idx == 0:
                # The first layer
                self.encoder.append(
                    nn.Conv2d(in_channels, feature, kernel_size=4,stride=2,padding=1)
                )
            else:
                # rest of the layers
                self.encoder.append(
                    DownSample(in_channels, feature, negative_slope = 0.2)
                )

            in_channels = feature

        features = features[::-1]
        # Decoder
        for idx in range(len(features) - 1):
            self.decoder.append(
                    UpSample(in_channels=features[idx]*2, out_channels=features[idx+1])
                )
            

    def size_check(self,x,size):
        b,c,h,w =x.shape
        if x.shape[-2:] != size:
            h,w = size
            x = TF.resize(x,(b,c,h,w))
        return x
        

    def forward(self,x):
        skip_connections = []
        # Down
        for idx,encode in enumerate(self.encoder):
            x = encode(x)
            # x = self.size_check(x,size[idx])
            skip_connections.append(x)

        # botle_neck
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        # Up
        for idx,decode in enumerate(self.decoder):
            # make sure skip connection is the same size as x
            if skip_connections[idx].shape[-2:] != x.shape[-2:]:
                x = TF.resize(x,skip_connections[-1].shape[-2:])

            concated_x = torch.concatenate((x,skip_connections[idx]),dim=1)
            x = decode(concated_x)

        # final layer
        if skip_connections[-1].shape[-2:] != x.shape[-2:]:
                x = TF.resize(x,skip_connections[-1].shape[-2:])
                

        concated_x = torch.concatenate((x,skip_connections[-1]),dim=1)
        return self.final(concated_x)






def test():
    # test input 256*192*22 and 256* 192*3
    p = torch.randn((3,22,256,192))
    c = torch.randn((3,3,256,192))
    input = torch.concatenate([p,c],dim = 1)
    output = torch.randn((3,4,256,192))

    gen = Generator()
    result = gen(input)
    if result.shape == output.shape:
        print("Test success")
    else:
        print("test fail")
        print("input shape: ",input.shape)
        print("result shape: ",result.shape)
        print("ideal output shape: ", output.shape)

if __name__ == "__main__":
    test()
    
