import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import albumentations as A
import torch


# Dataset - это абстрактный класс PyTorch, который обязывает нас реализовать три метода: __init__ __len__ __getitem__


class ParkingDataset(Dataset):
    def __init__(self, dataset_path, transform=None, mode = "train", test_size=0.2, random_seed=42):
        self.dataset_path = dataset_path
        self.transform = transform # для аугментации
        self.mode = mode
        self.samples = [] # список для хранения (путь к изображению, метка)
        self.class_names = ['free', 'occupied']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)} # Создает словарь {'free': 0, 'occupied': 1}
        self._load_and_balance_data(test_size=test_size, random_state=random_seed)

    def _load_and_balance_data(self, test_size, random_state):
        all_samples = []

        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name) # создаем полный путь к папке класса
            if not os.path.exists(class_path):
                print(f"Warning, {class_name} is not present.")
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)
                    all_samples.append((img_path, self.class_to_idx[class_name]))
        if not all_samples:
            raise FileNotFoundError("No images found.")

        # test size - доля данных для валидационной выборки (например 0.2)
        # random_state - фиксирует случайное разделение для воспроизводимости результатов
        train_samples, val_samples = train_test_split(all_samples, test_size = test_size, random_state=random_state, stratify= [label for _, label in all_samples])
        self.samples = train_samples if self.mode == "train" else val_samples

    def __len__(self):
        return len(self.samples)

    # image, label, path = dataset[5]
    # # Вызывается этот __getitem__ метод

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError("Image {} not found.".format(img_path))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Original shape: {image.shape}")  # (H, W, C)

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
                print(f"After transform: {image.shape}")  # (C, H, W)

            else:
                image = torch.from_numpy(image).permute(2,0,1).float() / 255.0 # преобразуем массив numpy в тензор PyTorch / меняем оси / во float и преобразовали от 0 до 1
                print(f"After basic processing: {image.shape}")  # (C, H, W)

            return image, label, img_path
        except FileNotFoundError:
            '''В больших датасетах могут быть битые или отсутствующие файлы'''
            print(f"Warning, {img_path} not found.")
            dummy_image = torch.zeros((3, 224, 224)) if self.transform else torch.zeros((3, 64, 64)) # Создаем "заглушку" чтобы обучение продолжалось
            return dummy_image, label, "error"


def get_transforms(input_size=224):
    """
    Создает трансформации для аугментации данных
    """
    train_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(input_size, input_size),
        # стандартные значения ImageNet значения стали стандартом де-факто
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def create_data_loaders(dataset_path, batch_size=32, input_size=224, num_workers=4):

    """
    Создает DataLoader'ы для тренировки и валидации
    """

    train_transform, val_transform = get_transforms(input_size)

    # создание датасетов
    train_dataset = ParkingDataset(
        dataset_path=dataset_path,
        transform=train_transform,
        mode='train'
    )

    val_dataset = ParkingDataset(
        dataset_path=dataset_path,
        transform=val_transform,
        mode='val'
    )

    # Создание даталоадеров
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Перемешивание каждую эпоху, чтобы предотвратить запоминание
        num_workers=num_workers,
        pin_memory=True, # Ускорение передачи на GPU быстрая передача CPU -> GPU
        drop_last=True  # Убираем неполные батчи для стабильности обучения -- Игнорирует неполный последний батч
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.class_names

# Пример использования
if __name__ == "__main__":
    DATASET_PATH = "/data/dataset_parking/parking_dataset"

    try:
        train_loader, val_loader, class_names = create_data_loaders(
            dataset_path=DATASET_PATH,
            batch_size=16,
            input_size=224
        )

        print("DataLoaders created")
        print(f"Classes: {class_names}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Train loaders: {train_loader}")
        print(f"Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Check your dataset path and structure!")