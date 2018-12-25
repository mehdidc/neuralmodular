import numpy as np
import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Gen(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, act='tanh', w=64):
        super().__init__()
        self.act = act
        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ngf * 2**(nb_blocks + 1)
        layers = [
            nn.ConvTranspose2d(nz, nf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ]) 
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out


class Discr(nn.Module):

    def __init__(self, nc=1, ndf=64, act='sigmoid', no=1, w=64):
        super().__init__()
        self.act = act
        self.no = no

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        layers.append(
            nn.Conv2d(nf, no, 4, 1, 0, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        out = self.main(input)
        if self.act == 'tanh':
            out = nn.Tanh()(out)
        elif self.act == 'sigmoid':
            out = nn.Sigmoid()(out)
        return out.view(-1, self.no)
