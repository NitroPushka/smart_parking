import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.dataloader import create_data_loaders
from src.classification.parking_classifier import ParkingClassifier
import torch.nn as nn
import time
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device, epoch = 50):
    """
    Полная функция обучения модели
    :param model: Обучающая модель
    :param train_loader: Данные для обучения
    :param val_loader: Данные для проверки
    :param device: Действующее устройство
    :param epoch: Количество эпох
    :return: train_losses, val_losses, train_accuracies, val_accuracies
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []


    errors = nn.CrossEntropyLoss() # функция потерь "Насколько далеки предсказания от правильных ответов"

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-4)  # оптимизатор градиентного спуска
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor= 0.5) # автопилот для скорости обучения - шагам LR

    start_time = time.time()
    for epochs in range(epoch):

        """ Обучение модели """
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = errors(outputs, labels)

            #backword pass
            loss.backward() # вычисление градиентов - в какоую сторону двигаться
            optimizer.step() # обновление весов - делает шаг в правильном направлении

            # Статистика

            predictions = torch.argmax(outputs, 1) # [0,1] - согласно показателям уверенности выбираем 0 (свободно) или 1 (занято)

            # Считаем точность
            # (predictions == labels) где наша модель угадала точно и считаем сумму
            correct_predictions = (predictions == labels).sum().item()
            total_images = labels.size(0) # кол-во изображений в текущем батче
            batch_accuracy = correct_predictions / total_images

            epoch_train_loss += loss.item() * total_images
            epoch_train_correct += correct_predictions
            epoch_train_total += total_images

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item():.3f}, Accuracy {batch_accuracy:.3f}')

        average_train_loss = epoch_train_loss / epoch_train_total
        average_train_accuracy = epoch_train_correct / epoch_train_total
        train_losses.append(average_train_loss)
        train_accuracies.append(average_train_accuracy)

        """ Валидация модели """
        model.eval() # Режим оценки / Отключает Dropout - использует ВСЕ нейроны / Фиксирует BatchNorm - использует накопленную статистику

        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad(): # отключаем вычисление градиентов
            for images, labels, paths in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Только forward pass (без обучения)
                outputs = model(images)
                loss = errors(outputs, labels)

                # Статистика валидации
                predictions = torch.argmax(outputs, 1)  # там гле высокая вероятность свободно то 0, иначе 1
                correct_predictions = (predictions == labels).sum().item() # сумма где предикт совпал с реальным
                total_images = labels.size(0) # кол-во изображении тек. батча

                epoch_val_loss += total_images * loss.item() # общая ошибка = количество изображений * на ошибку одного изображения
                epoch_val_correct += correct_predictions
                epoch_val_total += total_images

        average_val_loss = epoch_val_loss / epoch_val_total
        average_val_accuracy = epoch_val_correct / epoch_val_total

        val_accuracies.append(average_val_accuracy)

        # Фиксирует BatchNorm - использует накопленную статистику
        scheduler.step(average_val_loss) # обновление LR - здесь нам идет подсказка когда изменять LR для след. обучения

        print(f"\n Эпоха {epochs+1}/{epoch}")
        print(f" Обучение ==  Loss: {average_train_loss:.3f}, Accuracy: {average_train_accuracy:.3f} ")
        print(f"  Валидация == Loss: {average_val_loss:.3f}, Accuracy: {average_val_accuracy:.3f} ")
        print(f" Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        print("\n" + "="*50)

    training_time = time.time() - start_time
    print(f"Обучение модели завершено за {training_time}")

    print(f"Лучшая точность валидации: {max(val_accuracies)}")
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot(train_losses, val_losses, train_accuracies, val_accuracies):
    """ Построение графиков из полученных результатов обучения """
    plt.figure(figsize=(12, 4))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label = "Обучение", color = "red")
    plt.plot(val_losses, label = "Валидация", color = "blue")
    plt.title("Ошибки обучения модели")

    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label = "Обучение", color = "red")
    plt.plot(val_accuracies, label = "Валидация", color = "blue")
    plt.title("Точность предсказании модели")

    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()

def save_model(model, path = "parking_classifier.pth"):
    """Сохранение обученной модели"""
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в путь {path}")

def main():
    """ Обучение классификатора """

    DATASET_PATH = "/data/dataset_parking/parking_dataset"
    BATCH_SIZE = 32
    EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Задействовано устройство: {device}")

    try:

        train_loader, val_loader, class_names = create_data_loaders(DATASET_PATH, BATCH_SIZE, input_size= 224)
        print(f"Классы: {class_names}")
        print(f"Батчей для обучения: {len(train_loader)}")
        print(f"Батчей для валидации: {len(val_loader)}")

        print(f"Создание модели")
        model = ParkingClassifier(num_classes=len(class_names))
        model = model.to(device) # перекидываем на наше устройство -- скорее всего GPU

        print("Запуск обучения модели")
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, device, EPOCHS)
        save_model(model, '../../models/parking_classifier.pth')

        print("Строим графики")
        plot(train_losses, val_losses, train_accuracies, val_accuracies)

    except Exception as e:
        print(f"Внимание! Ошибка: {e}")

if __name__ == "__main__":
    main()
