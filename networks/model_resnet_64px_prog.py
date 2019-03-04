# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F
from .spectral_normalization import SpectralNorm
from .model_resnet import (ResBlockGenerator,
                          ResBlockDiscriminator,
                          FirstResBlockDiscriminator,
                          CBN2d)
                          
import numpy as np

class Generator(nn.Module):
    def __init__(self, nf, z_dim):
        super(Generator, self).__init__()
        self.nf = nf
        self.z_dim = z_dim
        
        self.dense = nn.Linear(self.z_dim, 4 * 4 * nf)
        self.final = nn.Conv2d(nf // 8, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        
        # 512, 256, 128, 64
        self.rbn1 = ResBlockGenerator(nf, nf,
                                      stride=2)
        self.rbn2 = ResBlockGenerator(nf, nf // 2,
                                      stride=2)
        self.rbn3 = ResBlockGenerator(nf // 2, nf // 4,
                                      stride=2)
        self.rbn4 = ResBlockGenerator(nf // 4, nf // 8,
                                      stride=2)
        #self.bn = CBN2d(y_dim, nf // 8)
        self.bn = nn.BatchNorm2d(nf // 8)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        #return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        x = self.dense(z).view(-1, self.nf, 4, 4)
        x = self.rbn1(x)
        x = self.rbn2(x)
        x = self.rbn3(x)
        x = self.rbn4(x)
        #x = self.bn(x, y)
        x = self.bn(x)
        x = self.relu(x)
        x = self.final(x)
        x = self.tanh(x)
        return x

    
class Discriminator(nn.Module):
    def __init__(self, nf,
                 sigmoid=False,
                 spec_norm=False):
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x
            
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(3, nf,
                                       stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*8, nf*8,
                                  spec_norm=spec_norm),
            nn.ReLU(),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Linear(nf*8, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        if self.sigmoid:
            return F.sigmoid(result)
        else:
            return result
        
def get_network(z_dim, sigm=True):
    gen = Generator(z_dim=z_dim,
                    nf=512)
    disc = Discriminator(nf=64,
                         sigmoid=sigm,
                         spec_norm=True)
    print("Generator:")
    print(gen)
    print("Discriminator:")
    print(disc)
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
