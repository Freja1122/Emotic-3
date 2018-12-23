import pickle
import os
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

category2num = {
    'Peace': 1,
    'Affection': 2,
    'Esteem': 3,
    'Anticipation': 4,
    'Engagement': 5,
    'Confidence': 6,
    'Happiness': 7,
    'Pleasure': 8,
    'Excitement': 9,
    'Surprise': 10,
    'Sympathy': 11,
    'Doubt/Confusion': 12,
    'Disconnection': 13,
    'Fatigue': 14,
    'Embarrassment': 15,
    'Yearning': 16,
    'Disapproval': 17,
    'Aversion': 18,
    'Annoyance': 19,
    'Anger': 20,
    'Sensitivity': 21,
    'Sadness': 22,
    'Disquietment': 23,
    'Fear': 24,
    'Pain': 25,
    'Suffering': 26
}

path = 'datasets/emotic'
with open('Annotations.pkl', 'rb') as f:
    annotations = pickle.load(f)
with open('Mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Emotic(Dataset):
    def __init__(self, part):
        self.part = part
        self.info = annotations[part]

    def __getitem__(self, item):
        j, k = mapping[self.part][item]
        info = self.info[j]
        filename = info['filename']
        folder = info['folder']
        image = Image.open(os.path.join(path, folder, filename)).convert('RGB')
        person = info['people'][k]
        left, top, right, bottom = person['body_box']
        body = image.crop((left, top, right, bottom))
        image, body = transform(image), transform(body)
        #categories = person['categories' if self.part == 'train' else 'combined_categories']
        if self.part == 'train':
            categories = person['categories']
        else:
            categories = person['categories'][0]
        dis_label = torch.zeros(26)
        for category in categories:
            dis_label[category2num[category] - 1] = 1
        return image, body, dis_label

    def __len__(self):
        return len(mapping[self.part].keys())
