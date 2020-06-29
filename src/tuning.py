
import itertools

import pandas as pd


def grid_search(run_func, configs):
    """
    Depricated: Personellement, je n'ais plus besoin de cette fonction; si vous en avez besoiin indique le.
    """
    
    results = []
    columns_name = None
    
    params = list(configs.keys())
    # Obtient les valeurs dans le même ordre que la liste des paramètres.
    values = list(configs[param] for param in params)
    # Combinaison possibles des paramêtres
    combinations = list(itertools.product(*values))
    
    for combination in combinations:
        trial_config = {}

        for iparam, value in enumerate(combination):
            trial_config[params[iparam]] = value

        # output est une liste de tuples (nom_param, valeur)
        output = run_func(trial_config)

        if columns_name is None:
            columns_name = ['config']
            # Initialise la liste des colones selon les premiers résultats
            columns_name.extend([param[0] for param in output])
            
        trial_result = [trial_config]
        trial_result.extend([param[1] for param in output])
        results.append(trial_result)
        
    return pd.DataFrame(results, columns=columns_name)
