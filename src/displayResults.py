# -*- coding: utf-8 -*-
"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 3 - Machine a vecteur de support et Reseaux de neuronnes

Students :
    Alexandre Laroche - LARA12078907
    Marc-Antoine Charland - CHAM16059609
    Jonathan Croteau-Dicaire - CROJ10109402

Group :
    GTI770-É19-02
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


class DisplayResults:

    def svm_linear_table(self, C, train_accuracy, test_accuracy, train_f1, test_f1, fit_time):
        """
        Fonction qui permet d'afficher le tableau des meilleurs résultats 
        du Grid Search SVM lineaire. 

        args:
            C:          liste des hyperparametres
            accuracy:   liste des meilleurs resultats
            fit_time:   liste du temps moyen du fit

        returns:
            Le tableau des hyperparametres SVM lineaire
        """
        # conversion les matrices en listes puis en une matrice
        results = np.column_stack([train_accuracy.tolist(), test_accuracy.tolist(), train_f1.tolist(), test_f1.tolist(), fit_time.tolist()])

        # Colonne principale (parametres)
        df = pd.DataFrame({'C': C})

        # Colonnes des resultats (precision et fitting time)
        df = pd.concat([df, pd.DataFrame(results, columns=['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1', 'Fitting Time (sec)'])], axis=1)

        # Precision des resultats (nombre de chiffres apres la virgule)
        pd.options.display.precision = 8

        # Ajustement de la taille et de l'alignement du tableau
        style = df.style
        style.set_properties(subset=['C'], **{'width': '50px', 'text-align': 'left'})
        style.set_properties(subset=['Train Accuracy'], **{'width': '100px', 'text-align': 'left'})
        style.set_properties(subset=['Test Accuracy'], **{'width': '100px', 'text-align': 'left'})
        style.set_properties(subset=['Train F1'], **{'width': '100px', 'text-align': 'left'})
        style.set_properties(subset=['Test F1'], **{'width': '100px', 'text-align': 'left'})
        style.set_properties(subset=['Fitting Time (sec)'], **{'width': '125px', 'text-align': 'left'})
        display(style)


    def svm_rbf_table(self, C, accuracy_fit):
        """
        Fonction qui permet d'afficher le tableau des meilleurs résultats ou
        fitting time du Grid Search SVM Non-lineaire (rbf).

        args:
            C:            liste des hyperparametres
            accuracy_fit: liste des resultats (accuracy ou fitting time)

        returns:
            Le tableau des resultats (accuracy ou fitting time) SVM Non-lineaire (rbf)
        """
        # conversion de la matrice en 4 listes
        split_list = accuracy_fit.tolist()
        results = [split_list[x:x+4] for x in range(0, len(split_list),4)]

        # Colonne principale (parametres)
        df = pd.DataFrame({'C/sigma': C})

        # Colonnes des resultats (precision et fitting time)
        df = pd.concat([df , pd.DataFrame(results, columns=C)], axis=1)

        # Precision des resultats (nombre de chiffres apres la virgule)
        pd.options.display.precision = 8

        # Ajustement de la taille et de l'alignement du tableau
        style = df.style
        style.set_properties(subset=['C/sigma'], **{'width':'60px','text-align': 'left'})
        style.set_properties(subset=C, **{'width':'80px','text-align':'left'})
        display(style)
