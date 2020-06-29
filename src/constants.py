# -*- coding: utf-8 -*-
"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 2 - Arbres de décision, Bayes naïf et KNN

Students :
    Alexandre Laroche - LARA12078907
    Marc-Antoine Charland - CHAM16059609
    Jonathan Croteau-Dicaire - CROJ10109402

Group :
    GTI770-É19-02
"""

"""
Ce fichier contient les constantes globale.

Pour les chemins des fichiers, veuillez en faire des chemins relatifs au root du projet.
"""

from pathlib import Path

PROJECT_ROOT_PATH = str(Path(__file__).parent.parent)

DATA_PATH = "data/"
PROCESSED_DATA_PATH = DATA_PATH  + "processed/"
PROCESSED_IMG_PATH = PROCESSED_DATA_PATH + "imgs/"
GALAXY_FEATURES_FILE_PATH = PROCESSED_DATA_PATH + "galaxy_feature_vectors.csv"
SPAM_FEATURES_FILE_PATH = PROCESSED_DATA_PATH + "spam.csv"
IMGS_LABELS_FILE_PATH = PROCESSED_DATA_PATH + "galaxy_label_data_set.csv"
RAW_DATA_PATH = DATA_PATH + "raw/"
RAW_IMG_PATH = RAW_DATA_PATH + "imgs/"
LOGS_PATH = "logs/"
CNN_LOGS_PATH = LOGS_PATH + "cnn/"
MLP_LOGS_PATH = LOGS_PATH + "mlp/"
SVM_LOGS_PATH = LOGS_PATH + "svm/"
MODELS_PATH = "models/"
CNN_MODELS_PATH = MODELS_PATH + "cnn/"
MLP_MODELS_PATH = MODELS_PATH + "mlp/"
SVM_MODELS_PATH = MODELS_PATH + "svm/"

