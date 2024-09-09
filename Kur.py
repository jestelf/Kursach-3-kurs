from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Путь к файлу аннотаций для обучающего или валидационного набора
annotation_file = 'F:\\git\\Kursach\\annotations\\instances_val2017.json'

# Загрузка набора данных COCO
coco = COCO(annotation_file)

# Получение всех категорий
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_names)))

# Визуализация примера изображения с аннотациями
img_ids = coco.getImgIds(catIds=coco.getCatIds(cat_names[0]))
img_data = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]

# Загрузка изображения
img = cv2.imread('F:\\git\\Kursach\\val2017\\' + img_data['file_name'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Загрузка аннотаций
annIds = coco.getAnnIds(imgIds=img_data['id'], catIds=coco.getCatIds(), iscrowd=None)
anns = coco.loadAnns(annIds)

# Визуализация изображения с наложенными аннотациями
plt.imshow(img)
coco.showAnns(anns)
plt.show()
