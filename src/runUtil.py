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

from collections import OrderedDict
import json
import os
from typing import List, Dict, Any

import numpy as np

import src.constants as constants

class NumpyDtypeJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32): 
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)
    
def resume_history(history):
    """
    Résume l'historique d'exécution d'un modèle Keras en extractant les métriques de la meilleur époque.
    
    Args:
        history: 
    Returns: Un OrderedDict contenant les métriques de la meilleur époque.
        
    """
    
    best_run_i = np.argmin(history['val_loss'])
    precision = history['precision'][best_run_i]
    recall = history['recall'][best_run_i]
    val_precision = history['val_precision'][best_run_i]
    val_recall = history['val_recall'][best_run_i]
    best_f1 = 2 * ((precision * recall) / (precision + recall))
    best_val_f1 = 2 * ((val_precision * val_recall) / (val_precision + val_recall))
    
    return OrderedDict([
        ('best_epoch', best_run_i + 1),
        ('best_loss', history['loss'][best_run_i]),
        ('best_val_loss', history['val_loss'][best_run_i]),
        ('best_accuracy', history['acc'][best_run_i]),
        ('best_val_accuracy', history['val_acc'][best_run_i]),
        ('best_f1', best_f1),
        ('best_val_f1', best_val_f1)
    ])

def save_history(history: np.ndarray, run_name: str):
    
    history_save_path = os.path.join(
        constants.PROJECT_ROOT_PATH,
        constants.CNN_LOGS_PATH,
        run_name + '_history'
    )
    
    np.save(history_save_path, history)

def save_run_results(run_results, run_name: str):

    save_path = os.path.join(
        constants.PROJECT_ROOT_PATH,
        constants.CNN_LOGS_PATH,
        run_name + '_run_results.json'
    )
    
    with open(save_path, 'w') as file:
        json.dump(run_results, file, cls=NumpyDtypeJSONEncoder)
    
def save_config(config: Dict[str, List[Any]], run_name: str):
    
    config_save_path = os.path.join(
        constants.PROJECT_ROOT_PATH,
        constants.CNN_LOGS_PATH,
        run_name + '_config.json'
    )
    
    with open(config_save_path, 'w') as file:
        json.dump(config, file)
