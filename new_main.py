import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32
img_height = 180
img_width = 180

def visualize(ds):
    import itertools
    # Visualisation
    plt.figure(figsize=(10, 10))
    for images, labels in itertools.islice(ds, 9):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].astype("uint8"))
            plt.title("class_names[labels[i]]")
            plt.axis("off")
        plt.show()

def show_train_result(history_, epochs_):
    acc = history_.history['accuracy']
    val_acc = history_.history['val_accuracy']

    loss = history_.history['loss']
    val_loss = history_.history['val_loss']

    epochs_range = range(epochs_)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def get_generator(df: pd.DataFrame, data_type: str, list_classes: list, preprocessing_function=preprocess_input):
    """Function to get an image generator

    Args:
        df (pd.DataFrame): jeu de données à utiliser
            Doit contenir les colonnes :
               - file_path : chemin vers une image
               - file_class : classe correspondante
        data_type (str) : type de données
            - 'train' si données entrainement
            - 'valid' si données de validation
            - 'test' si données de test
        list_classes (list): liste des classes à prédire
        preprocessing_function (?) : fonction de preprocessing à utiliser au chargement des images
        width (int): largeur de l'image une fois chargée
        height (int): hauteur de l'image une fois chargée
        batch_size (int): taille des batchs d'images générés par le generateur
    """

    # TODO : changer le preprocessing si vous voulez
    # TODO : changer la taille des images chargées si vous voulez
    # TODO : ajouter du data augmentation si vous voulez

    df = df.copy(deep=True)
    if data_type == 'train':
        data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)
        # Ajouter data augmentation ici si vous voulez
    else:
        data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)

    # Get generator
    if data_type != 'test':
        generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='file_class',
                                                       classes=list_classes,
                                                       target_size=(img_width, img_height), color_mode='rgb',
                                                       class_mode='categorical', validation_split=0.2, subset=data_type,
                                                       batch_size=batch_size, seed=124578, validate_filenames=False)
    # Pour jeu de test, on va se créer un faux dataframe avec une classe unique
    # L'idée est qu'on veut juste réappliquer le même preprocessing en entrée, on ne se sert pas de la classe puisqu'on veut juste faire une prédiction
    else:
        df['fake_class_col'] = 'all_classes'
        generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='fake_class_col',
                                                       classes=['all_classes'],
                                                       target_size=(img_width, img_height), color_mode='rgb',
                                                       class_mode='categorical',
                                                       batch_size=batch_size, seed=124578, validate_filenames=False)
    return generator

def load_data(csv_path : str, image_dir : str, subset : str):
    df = pd.read_csv(csv_path, sep=',')
    df = df[['new_id', 'label']].rename(columns={'new_id': 'filename', 'label': 'file_class'})
    available_files = os.listdir(image_dir)
    print("number of available files:", len(available_files))
    df['file_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    df['file_class'] = df['file_class'].apply(str)
    list_classes = list(df['file_class'].unique())
    # dict_classes = {i: v for i, v in enumerate(list_classes)}
    print("list_classes:", list_classes)
    ds = get_generator(df, subset, list_classes, preprocessing_function=None)
    return ds, list_classes

train_ds, class_names = load_data("train_labels.csv", "./images/train/train/", "training")
val_ds, _ = load_data("train_labels.csv", "./images/test/test/", "validation")

num_classes = len(class_names)

model = Sequential([
    layers.Input(shape=(img_width, img_height, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
show_train_result(history, epochs)

print("finished visualitation")