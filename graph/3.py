import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Указание пути к набору данных
data_dir = r'F:\git\Kursach'
data_type = 'val2017'
ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')
screenshot_dir = 'F:\git\Kursach/screen'

# Проверка существования директории для скриншотов
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Загрузка аннотаций
coco = COCO(ann_file)

# Получение всех изображений содержащих категорию 'person'
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)
images = coco.loadImgs(img_ids[:50])  # Загружаем первые 50 изображений для анализа

# Преобразование данных в DataFrame
data = []
for img in images:
    img_path = os.path.join(data_dir, data_type, img['file_name'])
    if not os.path.isfile(img_path):
        continue
    
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
    anns = coco.loadAnns(anns_ids)
    for ann in anns:
        data.append({
            'image_id': img['id'],
            'file_name': img['file_name'],
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

screenshot_dir
