"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 3 — Machines à vecteur de support et réseaux neuronaux

Students :
    Alexandre Laroche - LARA12078907
    Marc-Antoine Charland - CHAM16059609
    Jonathan Croteau-Dicaire - CROJ10109402

Group :
    GTI770-É19-02
"""

import cv2
import csv
import os
import src.constants as constants

def load_processed_imgs(max_load=None):

    imgs = []
    lbls = []

    with open(constants.IMGS_LABELS_FILE_PATH) as file:
        reader = csv.reader(file, delimiter=',')

        # skip the header
        next(reader)

        img_count = 0

        for row in reader:
            filename = row[0]
            label = row[1]

            filepath = os.path.join(os.path.abspath(''), constants.PROCESSED_IMG_PATH + '%s.jpg' %filename)

            if os.path.isfile(filepath):

                if max_load is None or img_count < max_load:
                    img = cv2.imread(filepath)
                    imgs.append(img)
                    lbls.append(label)
                
                    if max_load is not None :
                        img_count += 1

    return imgs, lbls
