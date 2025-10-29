import torch.nn as nn

class ParkingClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ParkingClassifier, self).__init__() # <- родительский констурктор .train()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate/2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate/2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate/2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # нормализация
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate/2) # случайно отключает 25% карт признаков во время обучения - предотвращает переобучение, заставляет сеть учить более крепкие признаки
        )

        """
        Берем среднее значение по всем пикселям - размерность HxW сжимается до 1 x 1
        [256] средних значений (по одному на канал)
        Необходим для универсальности к размеру входного изображения -> Не важно, какого размера было изображение на входе сети!
        Каждый канал становится одним числом - средним активацией по всей карте признаков -> сеть становится более устойчивой к небольшим смещениям и деформациям

        """
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        """
        Действуем по нисходящей размерности
        У нас есть 256 различных признаков после сверток - и мы с каждым слоем учимся комбинировать  признаки
        Постепенно сжимаем информацию, собирая комбинации признаков - при резком сжатии была бы потеря памяти
        
        256×128 + 128×64 + 64×2 = 32,768 + 8,192 + 128 = 41,088 параметров
        256×2 = 512 параметров (но работает хуже!)

        """

        self.classifier = nn.Sequential(
            # слой 1 - на вход идет [256, batch_size] после GAP

            nn.Dropout(dropout_rate), # Много признаков (256) -> нужно предотвратить созависимость нейронов
            nn.Linear(in_features=256,out_features=128), # комбинатор признаков
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # слой 2
            nn.Dropout(dropout_rate/2), # Уже отобраны важные комбинации признаков -> нужно сохранить больше информации для дальнейшего обучения
            nn.Linear(in_features=128,out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # выходной слой
            nn.Dropout(dropout_rate/4), # Остались самые важные признаки для классификации -> нужно сохранить максимум информации для точного решения
            nn.Linear(in_features=64, out_features=num_classes)
            # Логиты - инициальные прогнозы модели, которые ещё не нормализованы в вероятности
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """ Инициализация весов """
        for layer in self.modules(): #  это генератор, который возвращает все слои сети
            if isinstance(layer, nn.Conv2d): # работает со свёрточными слоями
                # Автоматически подбирает масштаб весов исходя из размеров слоя
                nn.init.xavier_uniform_(layer.weight) # веса распределены РАВНОМЕРНО _uniform_ [-scale, scale]
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0) # преобразуем смещения в 0
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01) # Создает веса с нормальным распределением mean = 0 std = 0.01
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Прямой проход через сеть
        :param x: Входной тензор [batch_size, channels, height, width]
        :return: логиты [batch_size, num_classes]

        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1) # [batch, 256] мы преобразуем из многомерного в одномерный вектор
        # -1 - означает "вычисли размер автоматически"

        x = self.classifier(x) # [batch, 2]
        return x
