import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

#Подготавливаем данные
transform = transforms.Compose([transforms.Resize(64,64),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

#ImageFolder — стандартный класс PyTorch, который автоматически подхватывает структуру папок:
train_data = datasets.ImageFolder("C:/Users/ilyap/PycharmProjects/smart_parking/data/dataset_parking/yolo_dataset/labels/train", transform=transform)
val_data = datasets.ImageFolder("C:/Users/ilyap/PycharmProjects/smart_parking/data/dataset_parking/yolo_dataset/labels/val", transform=transform)

# Batch_size = 16 — модель будет обрабатывать 16 изображений за раз.
# Shuffle = True — перемешиваем данные, чтобы модель не "запоминала" порядок.

# DataLoader делает итератор, который возвращает inputs, labels.
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

