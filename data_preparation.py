import numpy as np
from pycocotools.coco import COCO

def load_data(annotation_file):
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    np.random.shuffle(img_ids)
    train_size = int(0.8 * len(img_ids))
    val_size = int(0.1 * len(img_ids))

    train_ids = img_ids[:train_size]
    val_ids = img_ids[train_size:train_size + val_size]
    test_ids = img_ids[train_size + val_size:]

    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    train_ids, val_ids, test_ids = load_data('F:\\git\\Kursach\\annotations\\instances_val2017.json')
    print(f"Train IDs: {len(train_ids)}, Validation IDs: {len(val_ids)}, Test IDs: {len(test_ids)}")
