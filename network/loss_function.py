import torch 
import torch.nn as nn
import torchvision.models as models



class VGGloss(nn.Module):
    def __init__(self,layer_idx = [2,7,12,21,30],lambda_i = [1./32.,1./16.,1./8.,1./4.,1.],device = "cuda"):
        """
            Initialize the VGG loss
            Args:
                layer_idx: the index of the layers to calculate the loss
                lambda_i: the weight for each layer
                device: the device to run the loss
        """
        super(VGGloss,self).__init__()
        
        self.device = device    
        self.layer_idx = layer_idx
        self.lambda_i = lambda_i
        self.vgg = models.vgg19(pretrained = True).features.to(device)
        self.loss = nn.L1Loss()
        # self.to(device)



    def forward(self,I, I0):
        """
            Calculate the VGG loss between the two images
            Args:
                I: the generated image
                I0: the ground truth image
            Return:
                out: the VGG loss , a scalar
        
        """
        j = 0
        out = 0
        for i in range(len(self.vgg)):
            I = self.vgg[i](I)
            I0 = self.vgg[i](I0)

            if j >= len(self.layer_idx):
                break
            elif i == self.layer_idx[j]:
                out += self.loss(I,I0) * self.lambda_i[j]
                j += 1
        
        return out
    

def test():
    vgg = VGGloss()
    I = torch.randn(1,3,256,192).to("cuda")
    I0 = torch.randn(1,3,256,192).to("cuda")    
    loss = vgg(I,I0)


if __name__ == "__main__":
    test()