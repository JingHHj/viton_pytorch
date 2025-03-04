import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pdb 


class RefinementNet(nn.Module):
    def __init__(self,in_channels=6, out_channels=1,feature = 64):
        super(RefinementNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,feature,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(feature,feature,kernel_size=3,stride=1,padding=1)            
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(feature,out_channels,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()            
        )
        
        self.refine = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv2,
            self.conv3
        )

    def forward(self,wrapped_clothing, coarse_result):
        x = torch.concatenate((wrapped_clothing, coarse_result),dim=1)
        alpha = self.refine(x)
        refine_result = alpha * wrapped_clothing + (1 - alpha) * coarse_result

        return refine_result
    


def test():
    c_prime = torch.randn((10,3,256,192))
    I_prime = torch.randn((10,3,256,192))

    refinement = RefinementNet()
    result = refinement(c_prime, I_prime)
    print(result.shape)
        

if __name__ == "__main__":
    test()
