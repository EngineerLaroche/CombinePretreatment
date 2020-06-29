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

from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Any, Iterable


class ModelTrainer:

    NB_SPLITS = 10
    CLF_KEY = 'clf'
    CLF_KEY_PREFIX = 'clf__'

    def pipelined_search(self,
                         x: Iterable[Any],
                         y: Iterable[Any],
                         estimator: BaseEstimator,
                         parameters: Dict[str,List[Any]] = None,
                         transformers: List[Tuple[str, TransformerMixin]] = None,
                         test_size: float = None
                         ):
        """
        Effectue une recherche pour optimiser les paramètres de l'algorithme d'apprentissage 
        spécifié. Pour ce faire l'algorithme est entrainé avec chaque configuration possible. 
        La validation croisée du modèle est aussi effectuée. De plus, la méthode de validation 
        croisée utilisée dépend de la valeur du paramètre test_size. Enfin, des transformations 
        peuvent être appliquées à toutes les configurations testées.

        Args:
            x: Un array contenant les observations
            y: Un array contenant labels associées aux observations. L'encodage des labels est
                l'aissé à la discrétion de l'utilisateur de cette fonction.
            estimator: L'estimateur à entrainer.
            parameters: Un dictionnaire contenant les paramêtres à optimiser de l'estimateur. La
                clé doit être le nom du paramètre. La valeur associées à cette clé doit être une 
                liste contenant les valeurs possibles du paramètre.
            transformers: Une liste de tuple où le premier élément est une clé (string) associée
                au transformer. Le deuxième élément du tuple est le transformer. Ces transformers
                sont appliqués à toutes les configurations de l'algorithme.
            test_size: Si c'est un float entre 0 et 1, effectue une validation croisée avec la
                méthode Holdout où la proportion de l'ensemble de test est celle spécifié par 
                ce paramètre. Si test_size est None, effectue une validation croisée de type 
                K_fold stratifié où K = 10.
        
        Returns:
            Retourne le dictionnaire suivant :
            {
                "best_f1_score": Le score F1 de la meilleur configuration de l'algorithme
                "best_params": Un dictionnaire contenant les paramètres de la meilleur 
                    configuration de l'algorithme 
                "best_search_index": L'indexe de la meilleur configuration de l'algorithme. 
                    Cet indexe corresponds aux détails d'exécution de cette configurations dans 
                    les sous-array de cv_results.
                "cv_results": Dictionnaire qui contient les details et résultats de toutes les 
                    exécutions de toutes les configurations de l'algorithme telle que décrit par la 
                    documentation de pour l'attibut sklearn.model_selection.GridSearchCV.cv_results_
            }
        """

        pipeline_transformers = []

        if transformers is not None:
            pipeline_transformers.extend(transformers)

        pipeline_transformers.append((self.CLF_KEY, estimator))

        pipeline = Pipeline(pipeline_transformers)

        train_parameters = None

        if parameters is not None:
            train_parameters = {}

            for key, value in parameters.items():
                train_parameters[self.CLF_KEY_PREFIX + key] = value

        return self.__train_v(x, y, pipeline, train_parameters, test_size)

    def __train_v(self, x, y, estimator, parameters=None, test_size=None):

        """
        Effectue l'entrainement de l'estimateur spécifié, la validation croisée de celui-ci, 
        ainsi que la recherche des hyperparamètres optimales lors d'une recherche de type 
        Grid Search. Ainsi, l'algorithme d'apprentissage est entrainé pour chaque configuration 
        de paramêtre spécifié. De plus, la méthode de validation croisée utilisée dépend de la 
        valeur du paramètre test_size.
        
        Args:
            x: Un array contenant les observations
            y: Un array contenant labels associées aux observations. L'encodage des labels est
                l'aissé à la discrétion de l'utilisateur de cette fonction.
            estimator: L'estimateur à entrainer.
            parameters: Les paramêtres à optimiser pour l'estimateur. Correspond au paramêtre
                param_grid de sklearn.model_selection.GridSearchCV
            test_size: Si c'est un float entre 0 et 1, effectue une validation croisée avec la
                méthode Holdout où la proportion de l'ensemble de test est celle spécifié par 
                ce paramètre. Si test_size est None, effectue une validation croisée de type 
                K_fold stratifié où K = 10.
        
        Returns:
            Retourne le dictionnaire suivant :
            {
                "best_f1_score": Le score F1 de la meilleur configuration de l'algorithme
                "best_params": Un dictionnaire contenant les paramètres de la meilleur 
                    configuration de l'algorithme 
                "best_search_index": L'indexe de la meilleur configuration de l'algorithme. 
                    Cet indexe corresponds aux détails d'exécution de cette configurations dans 
                    les sous-array de cv_results.
                "cv_results": Dictionnaire qui contient les details et résultats de toutes les 
                    exécutions de toutes les configurations de l'algorithme telle que décrit par la 
                    documentation de pour l'attibut sklearn.model_selection.GridSearchCV.cv_results_
            }
        """
        
        f1_scorer = make_scorer(f1_score, average='weighted')
        accuracy_scorer = make_scorer(accuracy_score)

        if test_size is None:
            validator = StratifiedKFold(n_splits=self.NB_SPLITS, shuffle=True)
        elif test_size >= 1 or test_size <= 0:
            raise ValueError('test_size expected value must respect the boundaries ]0, 1[. ' +
                             'The value of test_size was : {}'.format(test_size))
        else:
            # Dans ce cas, on veut effectuer une validation holdout.
            x_train, x_test = train_test_split(x, test_size=test_size, shuffle=True, stratify=y)

            train_indexes = [x.index(elem) for elem in x_train]
            test_indexes = [x.index(elem) for elem in x_test]

            # cv of GridSearchCV as [(train_indexes, test_indexes)] a.k.a an iterable yielding
            # (train, test) splits as arrays of indices.
            validator = [(train_indexes, test_indexes)]

        grid_params = {}

        if parameters is not None:
            grid_params = parameters

        grid_search = GridSearchCV(
            estimator,
            grid_params,
            return_train_score=True,
            cv=validator,
            scoring={
                'accuracy': accuracy_scorer,
                'f1': f1_scorer
            },
            refit='f1',
            n_jobs=-1
        )
    
        grid_search.fit(x, y)

        results = {
            'best_f1_score': grid_search.best_score_,
            'best_params': self.rm_classifier_key_prefix(grid_search.best_params_),
            'best_search_index': grid_search.best_index_,
            'cv_results': grid_search.cv_results_
        }
        
        return results

    def rm_classifier_key_prefix(self, dict: Dict[str, Any]):
        """
        Retire le préfixe "clf__" des clés du dictionnaire spécifié.

        Args:
            parameters: Le dictionnaire don les clés seront amputées du préfixe "clf__"
        Returns:
            Le dictionnaire avec les clés amputées du préfixe "clf__"
        """

        result = {}

        for key, value in dict.items():

            key_len = len(self.CLF_KEY_PREFIX)

            if key.find(self.CLF_KEY_PREFIX, 0, key_len-1):
                key = key[key_len:]
                result[key] = value

        return result
