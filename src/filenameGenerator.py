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

import datetime
import os
import src.constants as constants


def next_cnn_savefilename():
    """
    Depricated:  Personellement, je n'ai plus besoin de cette fonction; si vous en avez besoin indiquez le.
    """
    
    date_part = _get_date_part()
    
    path = os.path.join(
        constants.PROJECT_ROOT_PATH,
        constants.CNN_MODELS_PATH,
        date_part + '_{}' + '.h5'
    )

    i = 1
    while os.path.exists(path.format(i)):
        i += 1

    save_name = date_part + '_{}'.format(i)
    file_path = path.format(i)

    return file_path, save_name

def _get_date_part():

    date = datetime.date.today()
    
    return date.strftime('%Y%m%d')
    
