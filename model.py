import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.classifier import ClassifierHead


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class EffiNet01(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)

#        for n, p in self.model.named_parameters():
#            if 'classifier' not in n:
#                p.requires_grad = False

        self.model.classifier.weight.data.normal_(0, 0.01)
        self.model.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        return x

class EffiNet03(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)

#        for n, p in self.model.named_parameters():
#            if 'classifier' not in n:
#                p.requires_grad = False

        self.model.classifier.weight.data.normal_(0, 0.01)
        self.model.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        return x

class EffiNet01MultiHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        n_features = self.model.num_features
        self.mask_classifier = ClassifierHead(in_chs=n_features, num_classes=3)
        self.gender_classifier = ClassifierHead(in_chs=n_features, num_classes=2)
        self.age_classifier = ClassifierHead(in_chs=n_features, num_classes=3)

    def forward(self, x):
        x = self.model.forward_features(x)
        z = self.age_classifier(x)
        y = self.gender_classifier(x)
        x = self.mask_classifier(x)

        return x, y, z
