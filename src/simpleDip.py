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

import numpy as np
import math

def center_crop(img, new_h=None, new_w=None):            
    """
    Fonction qui recadre une image en son centre.
    Supporte aussi les images non carré.

    Args:
        img:   OpenCV Image RGB 
        new_h: La nouvelle hauteur de l'image
        new_w: La nouvelle largeur de l'image  
    Returns:
        Image recadrée en son centre avec la nouvelle hauteur et largeur
    """   
    h = img.shape[0]
    w = img.shape[1]
    if new_w is None: new_w = min(w,h)
    if new_h is None: new_h = min(w,h)

    l = int(np.ceil((w - new_w) / 2))      # Left
    r = w - int(np.floor((w - new_w) / 2)) # Right
    t = int(np.ceil((h - new_h) / 2))      # Top
    b = h - int(np.floor((h - new_h) / 2)) # Bottom

    if len(img.shape) == 2: img_crop = img[t:b,l:r]
    else: img_crop = img[t:b,l:r,...]

    return img_crop
    
def draw_vignette(img,r):
    """ 
    Applique sur l'image une vignette solide; sans dégradé, en noir de rayon intérieur r.
    Une vignette correspond à l'assombrissement périphérique d'une photographie. En termes,
    plus simple, colore en noir l'extérieur du cercle centré de rayon r.

    Args:
        img: OpenCV Image RGB
        r:   Rayon du cercle en pixel
    Returns:
        Image avec une vignette solide noir de rayon intérieur r
    """  
    img_circle = img.copy()
    c = (img.shape[0]/2) - r

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            c_y = math.pow(i-c-r,2)
            c_x = math.pow(j-c-r,2)
            d = math.sqrt(c_x + c_y)           
            if(d > r): img_circle[i][j] = [0,0,0]

    return img_circle
