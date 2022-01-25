import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Input, Conv2D, Flatten,
                                     MaxPooling2D, Dropout)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam

from generator import get_generator

def get_classes_from_proba(predicted_proba, dict_classes):
    predicted_class = np.vectorize(lambda x: dict_classes[x])(predicted_proba.argmax(axis=-1))
    return predicted_class

def load_data(csv_path : str, image_dir : str, split=False):
    df = pd.read_csv(csv_path, sep=',')
    df = df[['new_id', 'label']].rename(columns={'new_id': 'filename', 'label': 'file_class'})
    available_files = os.listdir(image_dir)
    print("number of available files:", len(available_files))
    df['file_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    df['file_class'] = df['file_class'].apply(str)

    # Spécifique, on se limite à 3 types (Water, Fire & Grass)
    # df = df[df['file_class'].isin(['Water', 'Fire', 'Grass'])].reset_index(drop=True)

    if split:
        return train_test_split(df, test_size=0.2, stratify=df['file_class'])
    return df


def main():
    # On vérifie qu'on utilise bien un GPU :
    print(f"Utilisation GPU : {'OK' if len(tf.config.list_physical_devices('GPU')) > 0 else 'KO'}")

    df_train, df_valid = load_data("./train_labels.csv", "./images/train/train/", split=True)
    df_submit = load_data("./sample_submission.csv", "./images/test/test/")

    # On récupère la liste des classes à prédire
    list_classes = sorted(list(df_train['file_class'].unique()))
    dict_classes = {i: v for i, v in enumerate(list_classes)}
    print("list_classes:", list_classes)

    # On définit les générateurs
    width = 64
    height = 64
    batch_size = 32
    train_generator = get_generator(df_train, data_type='train', list_classes=list_classes,
                                    width=width, height=height, batch_size=min(batch_size, len(df_train)))
    valid_generator = get_generator(df_valid, data_type='test', list_classes=list_classes,
                                    width=width, height=height, batch_size=min(batch_size, len(df_valid)))
    learning_rate = 0.001  # TODO: à adapter à votre problème

    print("Generator Done")
    # Get input/output dimensions
    input_shape = (width, height, 3)  # 3 car format rgb dans nos générateurs
    num_classes = len(list_classes)

    # Input
    input_layer = Input(shape=input_shape)
    x = Rescaling(1./255)(input_layer)
    x = Conv2D(32, 3, padding='same', activation='elu', kernel_initializer="he_uniform")(x)
    x = MaxPooling2D(2, strides=2, padding='valid')(x)
    x = Conv2D(64, 3, padding='same', activation='elu', kernel_initializer="he_uniform")(x)
    x = MaxPooling2D(2, strides=2, padding='valid')(x)
    x = Conv2D(128, 3, padding='same', activation='elu', kernel_initializer="he_uniform")(x)
    x = MaxPooling2D(2, strides=2, padding='valid')(x)

    x = Flatten()(x)
    x = Dense(256, activation='elu', kernel_initializer="he_uniform")(x)
    out = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
    model = Model(inputs=input_layer, outputs=[out])

    # Set optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary du modèle
    model.summary()

    epochs = 100
    patience = 10

    # On train notre modèle (on y a ajoute un early stopping pour s'arrêter si plus de progression sur le jeu de valid)
    fit_history = model.fit(
        x=train_generator,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
        steps_per_epoch=len(df_train) // min(batch_size, len(df_train)),
        validation_data=valid_generator,
        validation_steps=len(df_valid) // min(batch_size, len(df_valid)),
        verbose=1,
        workers=8,
    )

    # Pour la partie prédiction, on veut un générateur mais sans mélange, ni data augmentation, etc...
    # Pour ça, on utilise l'argument data_type à 'test' (cf. définition de la fonction get_generator)
    data_train_to_test_generator = get_generator(df_train, data_type='test', list_classes=None,
                                                 width=width, height=height, batch_size=min(batch_size, len(df_train)))
    data_submit_to_test_generator = get_generator(df_submit, data_type='test', list_classes=None,
                                                 width=width, height=height, batch_size=min(batch_size, len(df_submit)))
    print("======= Prediction =======")
    predicted_proba_on_train = model.predict(data_train_to_test_generator, workers=8, verbose=1)
    train_classes = get_classes_from_proba(predicted_proba_on_train, dict_classes)
    predicted_proba_on_submit = model.predict(data_submit_to_test_generator, workers=8, verbose=1)
    submit_classes = get_classes_from_proba(predicted_proba_on_submit, dict_classes)
    print(len(predicted_proba_on_train) == df_train.shape[0])
    print(len(predicted_proba_on_submit) == df_submit.shape[0])

    print(submit_classes)
    print("=== Submission ===")
    print(len(submit_classes))
    submission = pd.read_csv("./sample_submission.csv", encoding="UTF8", sep=",")
    submission['label'] = submit_classes
    print(submission)
    submission.to_csv("submission.csv", encoding="utf8", sep=',', index=False)
    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
