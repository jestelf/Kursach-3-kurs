import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# Загрузка и предобработка данных
data_dir = pathlib.Path('F:\\git\\Kursach\\val2017\\')
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'), shuffle=True)
train_size = int(0.8 * len(list(list_ds)))
val_size = int(0.1 * len(list(list_ds)))
train_ds = list_ds.take(train_size)
val_ds = list_ds.skip(train_size).take(val_size)
test_ds = list_ds.skip(train_size + val_size)

def process_path(file_path):
    def _process_path(file_path):
        print("Processing file:", file_path)
        img_raw = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        if tf.reduce_any(tf.shape(img) == 0):
            return None
        img = tf.image.resize(img, [128, 128])
        img = tf.image.random_flip_left_right(img)
        return img

    img = tf.py_function(_process_path, [file_path], [tf.float32])
    img = img[0]  # tf.py_function wraps the output in a list
    img.set_shape([128, 128, 3])
    return img




def filter_valid_images(img):
    return img is not None

train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(filter_valid_images)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(filter_valid_images)
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(filter_valid_images)


print("Training dataset size:", len(list(train_ds)))
print("Validation dataset size:", len(list(val_ds)))
print("Test dataset size:", len(list(test_ds)))


# Создание и обучение модели
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Вычисление длины датасетов до применения `.repeat()`
train_size = len(list(train_ds))
val_size = len(list(val_ds))

# Обучение модели
history = model.fit(
    train_ds.repeat().batch(32),
    epochs=10,
    steps_per_epoch=train_size // 32,
    validation_data=val_ds.repeat().batch(32),
    validation_steps=val_size // 32
)

test_loss, test_acc = model.evaluate(test_ds.batch(32))
print('\nTest accuracy:', test_acc)
