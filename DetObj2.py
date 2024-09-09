import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from collections import Counter
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import skimage.io as io
import seaborn as sns
import time

# Путь к данным COCO
data_dir = r'F:\git\Kursach'
data_type = 'val2017'
ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')
screenshot_dir = 'screen'

# Проверка существования директории для скриншотов
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Загрузка аннотаций
coco = COCO(ann_file)

# Получение всех идентификаторов изображений
img_ids = coco.getImgIds()
img_infos = coco.loadImgs(img_ids[:50])  # Загружаем метаданные первых 50 изображений

# Функция для отображения изображения с аннотациями
def show_image_with_annotations(img, anns):
    plt.imshow(img)
    coco.showAnns(anns)
    plt.show()

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
def load_data(coco, img_ids):
    images = []
    labels = []
    img_infos = coco.loadImgs(img_ids)  # Загрузка метаданных изображений
    for idx, img_info in enumerate(img_infos):
        img_path = os.path.join(data_dir, data_type, img_info['file_name'])
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            
            ann_ids = coco.getAnnIds(imgIds=img_info['id'])
            anns = coco.loadAnns(ann_ids)
            img_labels = set()  # Используем множество, чтобы избежать дублирования меток для одного изображения
            for ann in anns:
                category_id = ann['category_id']
                img_labels.add(category_id)
            labels.append(list(img_labels))  # Преобразуем множество в список
        if idx % 10 == 0:  # Добавляем вывод отладочной информации
            print(f"Загружено изображений: {idx}")
    return images, labels

# Загрузка данных
images, labels = load_data(coco, img_ids[:50])  # Загружаем первые 50 изображений для примера (уменьшение размера данных)
# Печать количества изображений
print(f'Number of images: {len(images)}')

# Анализ распределения классов объектов
all_labels = [label for sublist in labels for label in sublist]
label_counter = Counter(all_labels)
num_classes = max(label_counter.keys()) + 1  # Учитываем все категории

# Визуализация распределения классов объектов
plt.figure(figsize=(10, 5))
plt.bar(label_counter.keys(), label_counter.values())
plt.xlabel('Class ID')
plt.ylabel('Frequency')
plt.title('Distribution of Object Classes in COCO Dataset')
plt.savefig(os.path.join(screenshot_dir, 'class_distribution.png'))
plt.show()

print("Label distribution in COCO dataset:")
for label, count in label_counter.items():
    print(f"Class ID {label}: {count} instances")

# Визуализация нескольких изображений из набора данных
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Исходный набор данных', fontsize=20)

for i, img_info in enumerate(img_infos[:9]):  # Используем img_info для получения метаданных
    img_path = os.path.join(data_dir, data_type, img_info['file_name'])  # Используем img_info['file_name']
    if os.path.isfile(img_path):
        image = io.imread(img_path)
        row, col = divmod(i, 3)
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
    else:
        print(f"Файл не найден: {img_path}")

plt.savefig(os.path.join(screenshot_dir, 'original_dataset.png'))
plt.show()

# Преобразование данных в DataFrame
data = []
for img_info in img_infos:  # Используем img_info вместо img
    img_path = os.path.join(data_dir, data_type, img_info['file_name'])
    if not os.path.isfile(img_path):
        continue
    
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        data.append({
            'image_id': img_info['id'],
            'file_name': img_info['file_name'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox']
        })

df = pd.DataFrame(data)

# Первичный анализ данных
df_description = df.describe(include='all')
print(df_description)

# Сохранение стандартных метрик в виде скриншота
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.text(0.5, 0.5, df_description.to_string(), horizontalalignment='center', verticalalignment='center', fontsize=12)
fig.savefig(os.path.join(screenshot_dir, 'data_metrics.png'))

# Проверка на наличие пропущенных значений
missing_data = df.isna().sum()
print(missing_data)

# Визуализация пропущенных значений
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.text(0.5, 0.5, missing_data.to_string(), horizontalalignment='center', verticalalignment='center', fontsize=12)
fig.savefig(os.path.join(screenshot_dir, 'missing_data.png'))

# Анализ аномалий в данных
label_counts = Counter(df['category_id'])
print("Label counts:", label_counts)

# Визуализация анализа аномалий
plt.figure(figsize=(10, 5))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('ID категории')
plt.ylabel('Частота')
plt.title('Распределение меток')
plt.savefig(os.path.join(screenshot_dir, 'label_distribution.png'))
plt.show()

# Очистка данных (удаление пропущенных значений для примера)
df_cleaned = df.dropna()

# Визуализация примера очищенных данных
df_cleaned_description = df_cleaned.describe(include='all')
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.text(0.5, 0.5, df_cleaned_description.to_string(), horizontalalignment='center', verticalalignment='center', fontsize=12)
fig.savefig(os.path.join(screenshot_dir, 'cleaned_data_metrics.png'))
plt.show()

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

# Формирование обучающей и тестирующей выборок
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_flat, test_size=0.2, random_state=42)

print(f"X_train:\n{X_train}")
print(f"y_train:\n{y_train}")
print(f"X_test:\n{X_test}")
print(f"y_test:\n{y_test}")

# Визуализация структуры Random Forest
def plot_random_forest_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Random Forest')
    ax.text(0.5, 0.5, 'Random Forest\nс множеством деревьев решений', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структуры Gradient Boosting
def plot_gradient_boosting_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Gradient Boosting')
    ax.text(0.5, 0.5, 'Gradient Boosting\nс последовательными деревьями решений', 
            horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структуры Stacking
def plot_stacking_structure(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Структура Stacking')
    layers = [
        "Random Forest (базовый алгоритм)",
        "Gradient Boosting (базовый алгоритм)",
        "Logistic Regression (финальный алгоритм)"
    ]
    for i, layer in enumerate(layers):
        ax.text(0.5, 1 - (i * 0.3), layer, 
                horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

# Визуализация структур моделей
plot_random_forest_structure('random_forest_structure.png')
plot_gradient_boosting_structure('gradient_boosting_structure.png')
plot_stacking_structure('stacking_structure.png')

# Проверка на наличие пропущенных значений
def check_missing_data(images, labels):
    missing_images = [i for i in images if i is None]
    missing_labels = [i for i in labels if i is None]
    print(f"Number of missing images: {len(missing_images)}")
    print(f"Number of missing labels: {len(missing_labels)}")

check_missing_data(images, labels)

# Проверка аномалий в метках
def check_anomalies(labels):
    label_counts = Counter([label for sublist in labels for label in sublist])
    print("Label counts:", label_counts)
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.title('Label Distribution')
    plt.show()

check_anomalies(labels)

# Понижение размерности с помощью PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_images)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=train_labels_flat, cmap='viridis')
plt.title("PCA of Image Data")
plt.colorbar()
plt.show()

# Понижение размерности с помощью t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(train_images)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=train_labels_flat, cmap='viridis')
plt.title("t-SNE of Image Data")
plt.colorbar()
plt.show()

# Кластеризация с помощью KMeans
kmeans = KMeans(n_clusters=min(num_classes, 10), random_state=42)
clusters = kmeans.fit_predict(train_images)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
plt.title("KMeans Clustering of Image Data")
plt.colorbar()
plt.show()

# Поиск аномалий с помощью DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
anomalies = dbscan.fit_predict(train_images)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=anomalies, cmap='viridis')
plt.title("DBSCAN Anomaly Detection")
plt.colorbar()
plt.show()

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

kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Уменьшаем количество разбиений до 3
model_accuracies = []
accuracies = []

for name, model in models:
    accuracies = []
    start_time = time.time()  # Замер времени обучения
    for train_index, test_index in kf.split(train_images):
        X_train, X_test = train_images[train_index], train_images[test_index]
        y_train, y_test = train_labels_flat[train_index], train_labels_flat[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    avg_accuracy = np.mean(accuracies)
    end_time = time.time()
    training_time = end_time - start_time  # Время обучения для модели
    model_accuracies.append((name, avg_accuracy, training_time))
    print(f"{name} Cross-Validation Accuracy: {avg_accuracy:.2f}, Training Time: {training_time:.2f} seconds")

# Определение цвета для каждой модели
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
model_colors = {}

# Создание графика
plt.figure(figsize=(10, 6))
for (name, accuracy, _), color in zip(model_accuracies, colors):
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

# Усовершенствование моделей (гиперпараметры, регуляризация)
# Пример Grid Search для Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=2)
rf_grid_search.fit(train_images, train_labels_flat)
print(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")
print(f"Best cross-validation accuracy for Random Forest: {rf_grid_search.best_score_:.2f}")

# Замеры времени обучения
start_time = time.time()
rf_model = RandomForestClassifier(**rf_grid_search.best_params_, random_state=42)
rf_model.fit(train_images, train_labels_flat)
end_time = time.time()
training_time = end_time - start_time
print(f"Training time for Random Forest: {training_time:.2f} seconds")

# Оценка модели Random Forest на тестовых данных
y_pred_rf = rf_model.predict(val_images)
rf_accuracy = accuracy_score(val_labels_flat, y_pred_rf)
rf_precision = precision_score(val_labels_flat, y_pred_rf, average='weighted', zero_division=0)
rf_recall = recall_score(val_labels_flat, y_pred_rf, average='weighted', zero_division=0)  # Добавление zero_division=0
rf_f1 = f1_score(val_labels_flat, y_pred_rf, average='weighted', zero_division=0)  # Добавление zero_division=0
print(f"Random Forest Test Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest Test Precision: {rf_precision:.2f}")
print(f"Random Forest Test Recall: {rf_recall:.2f}")
print(f"Random Forest Test F1 Score: {rf_f1:.2f}")

# Визуализация результатов (Матрица ошибок и ROC-кривые)
conf_matrix = confusion_matrix(val_labels_flat, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ROC-кривые для Random Forest
y_pred_proba_rf = rf_model.predict_proba(val_images)
fpr, tpr, _ = roc_curve(val_labels_flat, y_pred_proba_rf[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Оценка модели мультиклассовой логистической регрессии на тестовых данных
logreg_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
])
logreg_model.fit(train_images, train_labels_flat)
y_pred_logreg = logreg_model.predict(val_images)
logreg_accuracy = accuracy_score(val_labels_flat, y_pred_logreg)
logreg_precision = precision_score(val_labels_flat, y_pred_logreg, average='weighted', zero_division=0)
logreg_recall = recall_score(val_labels_flat, y_pred_logreg, average='weighted', zero_division=0)  # Добавление zero_division=0
logreg_f1 = f1_score(val_labels_flat, y_pred_logreg, average='weighted', zero_division=0)  # Добавление zero_division=0
print(f"Logistic Regression Test Accuracy: {logreg_accuracy:.2f}")
print(f"Logistic Regression Test Precision: {logreg_precision:.2f}")
print(f"Logistic Regression Test Recall: {logreg_recall:.2f}")
print(f"Logistic Regression Test F1 Score: {logreg_f1:.2f}")
