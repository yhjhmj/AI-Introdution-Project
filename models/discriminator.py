import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw, pad = 4, 1
        layers = [
            nn.Conv2d(in_ch, ndf, kw, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),
        ]
        nf = 1
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=2, padding=pad),
                nn.InstanceNorm2d(ndf * nf),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf
        nf = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_prev, ndf * nf, kw, stride=1, padding=pad),
            nn.InstanceNorm2d(ndf * nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf, 1, kw, stride=1, padding=pad),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
