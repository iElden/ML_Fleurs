import pandas as pd
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


def get_generator(df: pd.DataFrame, data_type: str, list_classes: list, preprocessing_function=preprocess_input,
                  width: int = 224, height: int = 224, batch_size: int = 64):
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
    # Copy

    df = df.copy(deep=True)
    if data_type == 'train':
        data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)
        # Ajouter data augmentation ici si vous voulez
    else:
        data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)

    # Get generator
    shuffle = True if data_type == 'train' else False  # Attention, ne pas shuffle si valid/test !!!!
    if data_type != 'test':
        generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='file_class',
                                                       classes=list_classes,
                                                       target_size=(width, height), color_mode='rgb',
                                                       class_mode='categorical',
                                                       batch_size=batch_size, shuffle=shuffle, validate_filenames=False)
    # Pour jeu de test, on va se créer un faux dataframe avec une classe unique
    # L'idée est qu'on veut juste réappliquer le même preprocessing en entrée, on ne se sert pas de la classe puisqu'on veut juste faire une prédiction
    else:
        df['fake_class_col'] = 'all_classes'
        generator = data_generator.flow_from_dataframe(df, directory=None, x_col='file_path', y_col='fake_class_col',
                                                       classes=['all_classes'],
                                                       target_size=(width, height), color_mode='rgb',
                                                       class_mode='categorical',
                                                       batch_size=batch_size, shuffle=False, validate_filenames=False)
    return generator
