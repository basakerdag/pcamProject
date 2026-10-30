import torch.nn as nn

def get_act(name):
    return {"relu": nn.ReLU, "leakyrelu": lambda: nn.LeakyReLU(0.01), "gelu": nn.GELU}[name]()

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(96 * 96 * 3, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, act="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 96 * 3, 512), get_act(act), nn.Dropout(0.3),
            nn.Linear(512, 256), get_act(act), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class SmallCNN(nn.Module):
    def __init__(self, act="relu", chans=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       get_act(act), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(c_in, 1))
    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

def build_model(name, act="relu"):
    if name == "logreg": return LogReg()
    if name == "mlp": return MLP(act)
    if name == "smallcnn": return SmallCNN(act)
    if name in ("resnet18", "efficientnet_b0"):
        return Squeeze(build_pretrained(name))
    raise ValueError(name)

class Squeeze(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x).squeeze(-1)

def build_pretrained(name, pretrained=True):
    import torchvision
    tv = torchvision.models
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


class AutoEncoder(nn.Module):
    def __init__(self, act="relu", chans=(32, 64, 128, 256)):
        super().__init__()
        self.encoder = SmallCNN(act, chans).features
        rev = list(reversed(chans))
        layers = []
        for i in range(len(rev)):
            c_out = rev[i + 1] if i + 1 < len(rev) else rev[-1]
            layers += [nn.ConvTranspose2d(rev[i], c_out, 2, stride=2), nn.BatchNorm2d(c_out), get_act(act)]
        layers += [nn.Conv2d(rev[-1], 3, 3, padding=1)]
        self.decoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.decoder(self.encoder(x))