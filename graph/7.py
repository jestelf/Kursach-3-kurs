import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Путь к данным COCO
dataDir = r'F:\git\Kursach'  # Обновите путь к директории с данными
dataType = 'val2017'
annFile = os.path.join(dataDir, f'annotations/instances_{dataType}.json')

# Загрузка аннотаций COCO
coco = COCO(annFile)

# Получение всех идентификаторов изображений
imgIds = coco.getImgIds()
images = coco.loadImgs(imgIds[:50])  # Загружаем первые 50 изображений для тестирования (уменьшение размера данных)

# Проверка наличия файла перед загрузкой
def load_image(img_path):
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Ошибка загрузки изображения: {img_path}")
    else:
        print(f"Файл не найден: {img_path}")
    return None

# Загрузка данных и преобразование bounding boxes
def load_data(coco, imgIds):
    images = []
    labels = []
    for idx, imgId in enumerate(imgIds):
        img_info = coco.loadImgs(imgId)[0]
        img_path = os.path.join(dataDir, dataType, img_info['file_name'])
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            
            annIds = coco.getAnnIds(imgIds=img_info['id'])
            anns = coco.loadAnns(annIds)
            img_labels = set()  # Используем множество, чтобы избежать дублирования меток для одного изображения
            for ann in anns:
                category_id = ann['category_id']
                img_labels.add(category_id)
            labels.append(list(img_labels))  # Преобразуем множество в список
        if idx % 10 == 0:  # Добавляем вывод отладочной информации
            print(f"Загружено изображений: {idx}")
    return images, labels

# Загрузка данных
images, labels = load_data(coco, imgIds[:50])  # Загружаем первые 50 изображений для примера (уменьшение размера данных)

# Анализ распределения классов объектов
all_labels = [label for sublist in labels for label in sublist]
label_counter = Counter(all_labels)
num_classes = max(label_counter.keys()) + 1  # Учитываем все категории

# Преобразование данных для обучения
train_images = np.array([cv2.resize(img, (224, 224)).flatten() for img in images[:40]])  # 40 обучающих изображений
val_images = np.array([cv2.resize(img, (224, 224)).flatten() for img in images[40:]])   # 10 валидационных изображений

# Создаем "one-hot" представление для меток
def create_one_hot_labels(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label_list in enumerate(labels):
        for label in label_list:
            one_hot_labels[i, label] = 1
    return one_hot_labels

train_labels = create_one_hot_labels(labels[:40], num_classes)  # 40 обучающих меток
val_labels = create_one_hot_labels(labels[40:], num_classes)    # 10 валидационных меток

# Конвертируем "one-hot" представление в плоские метки для sklearn
train_labels_flat = np.argmax(train_labels, axis=1)
val_labels_flat = np.argmax(val_labels, axis=1)

# Используем LabelEncoder для последовательного кодирования меток
le = LabelEncoder()
all_labels_flat = np.concatenate([train_labels_flat, val_labels_flat])  # Объединяем метки для обучения
le.fit(all_labels_flat)
train_labels_flat = le.transform(train_labels_flat)
val_labels_flat = le.transform(val_labels_flat)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_flat, test_size=0.2, random_state=42)

# Список моделей для обучения
models = [
    ("SVM", SVC(max_iter=1000)),
    ("KNN", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("Bagging", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)),
    ("MLP", MLPClassifier(max_iter=300))
]

accuracies = []

# Обучение моделей и вычисление точности
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((name, accuracy))

# Определение цвета для каждой модели
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
model_colors = {}

# Создание графика
plt.figure(figsize=(10, 6))
for (name, accuracy), color in zip(accuracies, colors):
    if accuracy in model_colors:
        model_colors[accuracy].append(name)
    else:
        model_colors[accuracy] = [name]

# Переопределение цветов для пересекающихся моделей
new_colors = {}
for idx, (accuracy, names) in enumerate(model_colors.items()):
    if len(names) > 1:
        new_color = plt.cm.hsv(idx / len(model_colors))
        new_colors[new_color] = names
        plt.plot(range(len(X_test)), [accuracy] * len(X_test), color=new_color, label=', '.join(names))
    else:
        plt.plot(range(len(X_test)), [accuracy] * len(X_test), color=colors[idx], label=names[0])

plt.xlabel('Итерации')
plt.ylabel('Точность')
plt.title('Точность предсказаний моделей на тестовом наборе данных')
plt.legend(loc='best')
plt.show()
