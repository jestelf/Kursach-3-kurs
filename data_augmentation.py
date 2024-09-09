import tensorflow as tf
import pathlib

# Путь к папке с изображениями
data_dir = pathlib.Path('F:\\git\\Kursach\\val2017\\')

# Создание датасета
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'), shuffle=True)

# Разделение на обучающую, валидационную и тестовую выборки
train_size = int(0.8 * len(list(list_ds)))
val_size = int(0.1 * len(list(list_ds)))

train_ds = list_ds.take(train_size)
val_ds = list_ds.skip(train_size).take(val_size)
test_ds = list_ds.skip(train_size + val_size)

# Функция для загрузки и аугментации изображения
def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.image.random_flip_left_right(img)  # Пример аугментации
    return img

# Применение функции к каждому элементу в датасете
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
