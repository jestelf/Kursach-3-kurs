import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io

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
images = coco.loadImgs(img_ids[:9])  # Загружаем первые 9 изображений для демонстрации

# Визуализация нескольких изображений из набора данных
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Исходный набор данных', fontsize=20)

for i, img in enumerate(images):
    img_path = os.path.join(data_dir, data_type, img['file_name'])
    if os.path.isfile(img_path):
        image = io.imread(img_path)
        row, col = divmod(i, 3)
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
    else:
        print(f"Файл не найден: {img_path}")

plt.savefig(os.path.join(screenshot_dir, 'original_dataset.png'))
plt.show()
