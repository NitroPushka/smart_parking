import os
import sys

import subprocess

def print_menu():
    print("\t ====== Добро пожаловать в основное меню Smart Parking Detection System ====== ")
    print("-"*50)
    print("\t Обучение (опционально)")
    print("1 - Обучение YOLO детектора с новыми весами")
    print("2 - Обучение CNN-классификатора с новыми весами")
    print("-"*50)
    print("\t Запуск готовых решении")
    print("3 - Запуск YOLO детектора с Shapely парковкой")
    print("4 - Запуск классификатора")
    print("5 - exit")

def training_detection():
    print("Запуск обучения YOLO детектора...")
    try:
        from scripts.learning.yolo_detector import train_model as train_model
        train_model()
    except Exception as e:
        print(f"Произошла ошибка при обучении детектора: {e}")
        return False
    return True

def training_cnn():
    print("Запуск обучения классификатора...")
    try:
        from scripts.learning.learning_classifier import main as train_model
        train_model()
    except Exception as e:
        print(f"Произошла ошибка при обучении классификатора: {e}")
        return False
    return True

def detector():
    print("Запуск YOLO детектора...")
    try:
        from scripts.detect_with_shapely import main as main_detector
        main_detector()
    except Exception as e:
        print(f"Произошла ошибка при обучении классификатора: {e}")
        return False
    return True

def classifier():
    print("Запуск CNN-классификатора...")
    try:
        from scripts.classifier_live import main as main_classifier
        main_classifier()
    except Exception as e:
        print(f"Произошла ошибка при обучении классификатора: {e}")
        return False
    return True

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    while True:
        print_menu()
        try:
            choice = int(input("Выберите режим работы (1-5): "))

            if choice == 1:
                training_detection()
            elif choice == 2:
                training_cnn()
            elif choice == 3:
                detector()
            elif choice == 4:
                classifier()
            elif choice == 5:
                print("До свидания!")
                break
            else:
                print("Неизвестное значение! Попробуйте снова!")
                break
            print("\n" + "="*50)
            cont = input("Вернуться в меню? (y/n): ").strip().lower()
            if cont != 'y':
                print("До свидания!")
                break
        except Exception as e:
            print(f"Ошибка выбора режима работы: {e}")
            break

if __name__ == "__main__":
    main()





