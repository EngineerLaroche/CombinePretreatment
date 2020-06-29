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

import csv
import cv2
import numpy as np
import os
import src.constants as constants
import src.simpleDip as dip

class RawImagePreprocessor:

    _GAUSSIAN_KERNEL = 7
    # Original size : 424x424
    _RESIZE_HEIGHT = 212
    _RESIZE_WIDTH = 212
    _AFTERCROP_HEIGHT = 212*(280/424)
    _AFTERCROP_WIDTH = 212*(280/424)
    _VIGETTE_RADIUS = _AFTERCROP_WIDTH/2

    def preprocess_raw_imgs(self, verbose=False):

        nb_proc_imgs = 0
        
        with open(constants.IMGS_LABELS_FILE_PATH) as file:
            reader = csv.reader(file, delimiter=',')
            
            # skip the header
            next(reader)
    
            for row in reader:
                filename = row[0]
                raw_filepath = os.path.join(os.path.abspath(''), constants.RAW_IMG_PATH + '%s.jpg' %filename)
                proc_filepath = os.path.join(os.path.abspath(''), constants.PROCESSED_IMG_PATH + '%s.jpg' %filename)

                if os.path.isfile(raw_filepath):
                    # The file exists; then process it
                    img = cv2.imread(raw_filepath)
                    img = self._preprocess_img(img)
                    cv2.imwrite(proc_filepath, img)

                    if verbose:
                        nb_proc_imgs += 1
                        print(str(filename) + '.jpg preprocessed; ' + str(nb_proc_imgs) + ' total images processed')

    def _preprocess_img(self, img):
        """ 
        Effectue tous les prétraitement sur l'image donnée.

        Args:
            img: L'image à prétraiter
        Returns:
            L'image prétraitée
        """

        img = cv2.resize(img, (self._RESIZE_HEIGHT, self._RESIZE_WIDTH))
        gaussian_img = cv2.GaussianBlur(img, (self._GAUSSIAN_KERNEL,self._GAUSSIAN_KERNEL), cv2.BORDER_WRAP)
        img = dip.center_crop(img, self._AFTERCROP_HEIGHT, self._AFTERCROP_WIDTH)
        img = dip.draw_vignette(img, self._VIGETTE_RADIUS)
        return img
