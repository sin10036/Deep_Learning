import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        def block(_in, _out, stride=2, padding=1, bn=True, last=False):
            blk = [nn.Conv2d(_in, _out, 4, stride=stride, padding=padding)]
            if not last:
                blk.append(nn.LeakyReLU(0.2))
            if bn:
                blk.append(nn.BatchNorm2d(_out))
            return blk

        self.model = nn.Sequential(
            *block(3, 128, bn=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 1, stride=1, padding=0, bn=False, last=True),
            #nn.Linear(16, 1)
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        
        ##########       END      ##########
        
        return out


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        def block(_in, _out, stride=2, padding=1):
            blk = [nn.ConvTranspose2d(_in, _out, 4, stride=stride, padding=padding),
                    nn.ReLU(),
                    nn.BatchNorm2d(_out)]
            return blk

        self.model = nn.Sequential(
            *block(noise_dim, 1024, stride=1, padding=0),
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            *block(128, 3),
            nn.Tanh()
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = x.view(-1, self.noise_dim, 1, 1)
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        
        ##########       END      ##########
        
        return out
        

