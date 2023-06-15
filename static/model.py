from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models import mobilenet_v2
from torch import nn

model = mobilenet_v2()
features = list(model.classifier.children())[:-1]
features.extend([nn.Linear(model.classifier[1].in_features, 2)])
layer = nn.Sequential(*features)


class ImageClassifier(MobileNetV2):
    def __init__(self):
        super().__init__()
        self.classifier = layer
