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

import csv

class FeatureExtractor:

    def get_pourriel_feature_vector(self, csv_path):
        """
        Retourne le vecteur de label et le vecteur de primitive de l'ensemble de données des pourriels.

        Returns:
            Un Array contennant le vecteur de label
            Un array contennant le vecteur de primitive
        """
        label = []
        primitive = []

        with open(csv_path, "r") as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                row_features = [float(feature) for feature in row[0:len(row)-1]]
                primitive.append(row_features)
                if row[len(row)-1] == "0":
                    label.append("false")
                else:
                    label.append("true")

        return label, primitive

    def get_galaxy_feature_vector(self, csv_path):
        """
        Retourne le vecteur de label et le vecteur de primitive de l'ensemble de données des galaxies.

        Returns:
            Un Array contennant le vecteur de label
            Un array contennant le vecteur de primitive
        """
        label = []
        primitive = []

        with open(csv_path, "r") as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                row_features = [float(feature) for feature in row[0:len(row) - 1]]
                primitive.append(row_features)
                if row[len(row)-1] == "0":
                    label.append("smooth")
                else:
                    label.append("spiral")

        return label, primitive
