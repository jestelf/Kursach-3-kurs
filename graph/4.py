import os
from pycocotools.coco import COCO
import pandas as pd

# Указание пути к набору данных
data_dir = r'F:\git\Kursach/Kursach'
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
images = coco.loadImgs(img_ids[:5])  # Загружаем первые 5 изображений для упрощения анализа

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

# Проверка данных перед созданием DataFrame
print("Sample data for DataFrame creation:")
for sample in data:  # Выводим все записи для проверки
    print(sample)

# Проверка структуры данных
print("Structure of data:", type(data), len(data), type(data[0]) if len(data) > 0 else "Empty")

# Проверка типов всех данных
for i, item in enumerate(data):
    print(f"Item {i}: {item}")
    for key, value in item.items():
        print(f"  {key}: {type(value)}")

# Создание DataFrame с принудительным указанием типов данных
df_full = pd.DataFrame.from_records(data, columns=['image_id', 'file_name', 'category_id', 'bbox'])

# Проверка наличия ключей и первых строк DataFrame
print("Available columns in DataFrame:", df_full.columns)
print("First few rows of DataFrame:\n", df_full.head())
