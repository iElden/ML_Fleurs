import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (ELU, BatchNormalization, Dense, Dropout, SpatialDropout2D,
                                     Input, LeakyReLU, ReLU, Conv2D, Flatten,
                                     MaxPooling2D, AveragePooling2D,
                                     GlobalMaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import SGD, Adam
from typing import Tuple, Any

from generator import get_generator

def load_data(csv_path : str, image_dir : str) -> Tuple[Any, Any, Any]:
    df = pd.read_csv(csv_path, sep=',')
    df = df[['new_id', 'label']].rename(columns={'new_id': 'filename', 'label': 'file_class'})
    available_files = os.listdir(image_dir)
    print(available_files)
    df['file_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    df['file_class'] = df['file_class'].apply(str)

    # Spécifique, on se limite à 3 types (Water, Fire & Grass)
    # df = df[df['file_class'].isin(['Water', 'Fire', 'Grass'])].reset_index(drop=True)

    df_train, df_valid = train_test_split(df, test_size=0.2, stratify=df['file_class'])
    return df_train, df_valid, df


def main():
    # On vérifie qu'on utilise bien un GPU :
    print(f"Utilisation GPU : {'OK' if len(tf.config.list_physical_devices('GPU')) > 0 else 'KO'}")

    df_train, df_valid, df = load_data("./train_labels.csv", "./images/train/train/")

    # On récupère la liste des classes à prédire
    list_classes = sorted(list(df['file_class'].unique()))
    dict_classes = {i: v for i, v in enumerate(list_classes)}
    print("list_classes:", list_classes)

    # On définit les générateurs
    width = 64
    height = 64
    batch_size = 32
    train_generator = get_generator(df_train, data_type='train', list_classes=list_classes,
                                    width=width, height=height, batch_size=min(batch_size, len(df_train)))
    valid_generator = get_generator(df_valid, data_type='valid', list_classes=list_classes,
                                    width=width, height=height, batch_size=min(batch_size, len(df_valid)))
    learning_rate = 0.001  # TODO: à adapter à votre problème

    # Get input/output dimensions
    input_shape = (width, height, 3)  # 3 car format rgb dans nos générateurs
    num_classes = len(list_classes)

    # Input
    input_layer = Input(shape=input_shape)

    # Feature extraction

    x = Conv2D(32, 3, padding='same', activation='elu', kernel_initializer="he_uniform")(input_layer)
    x = MaxPooling2D(2, strides=2, padding='valid')(x)

    x = Conv2D(32, 3, padding='same', activation='elu', kernel_initializer="he_uniform")(x)
    x = MaxPooling2D(2, strides=2, padding='valid')(x)

    # Flatten
    x = Flatten()(x)

    # Classification
    x = Dense(256, activation='elu', kernel_initializer="he_uniform")(x)

    # Last layer
    out = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)

    # Set model
    model = Model(inputs=input_layer, outputs=[out])

    # Set optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    # Summary du modèle
    model.summary()

    # Pour la partie prédiction, on veut un générateur mais sans mélange, ni data augmentation, etc...
    # Pour ça, on utilise l'argument data_type à 'test' (cf. définition de la fonction get_generator)
    data_train_to_test_generator = get_generator(df_train, data_type='test', list_classes=None,
                                                 width=width, height=height, batch_size=min(batch_size, len(df_train)))
    data_valid_to_test_generator = get_generator(df_valid, data_type='test', list_classes=None,
                                                 width=width, height=height, batch_size=min(batch_size, len(df_valid)))
    predicted_proba_on_train = model.predict(data_train_to_test_generator, workers=8, verbose=1)
    predicted_proba_on_valid = model.predict(data_valid_to_test_generator, workers=8, verbose=1)
    print(len(predicted_proba_on_train) == df_train.shape[0])
    print(len(predicted_proba_on_valid) == df_valid.shape[0])

if __name__ == '__main__':
    main()
