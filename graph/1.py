import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from collections import Counter

# Указание пути к набору данных
data_dir = r'F:\git\Kursach'
data_type = 'val2017'
ann_file = f'{data_dir}/annotations/instances_{data_type}.json'

# Загрузка аннотаций
coco = COCO(ann_file)

# Получение всех изображений содержащих категорию 'person'
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)
images = coco.loadImgs(img_ids)

# Печать количества изображений
print(f'Number of images: {len(images)}')

# Анализ структуры данных
all_labels = [ann['category_id'] for img_id in img_ids for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids))]
label_counter = Counter(all_labels)

# Визуализация распределения классов объектов
plt.figure(figsize=(10, 5))
plt.bar(label_counter.keys(), label_counter.values())
plt.xlabel('Class ID')
plt.ylabel('Frequency')
plt.title('Distribution of Object Classes in COCO Dataset')
plt.savefig('class_distribution.png')
plt.show()
