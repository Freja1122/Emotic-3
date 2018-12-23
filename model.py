import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.body_feature_extraction = resnet50(pretrained=True)
        self.body_feature_extraction.fc = nn.Linear(2048, 1000)
        self.image_feature_extraction = resnet50()
        self.image_feature_extraction.fc = nn.Linear(2048, 365)
        self.fc = nn.Linear(1365, 1000)
        self.discrete = nn.Linear(1000, 26)
        self.continuous = nn.Linear(1000, 3)

    def forward(self, body, image):
        body_feature = self.body_feature_extraction(body)
        image_feature = self.image_feature_extraction(image)
        feature = F.relu(torch.cat((body_feature, image_feature), dim=1))
        feature = F.relu(self.fc(feature))
        return F.sigmoid(self.discrete(feature)), F.sigmoid(self.continuous(feature))
