import numpy as np
import os
import sys
from settings import params, fix_settings

def evaluate_relationship_prediction(predictions, targets):
    res = {'metric': 0., 'metric1': 0., 'metric': 0.}
    return res


def main(params, which_expr):
    # load predictions
    predictions = np.load(os.path.join('results', params['dataset'], 'predictions.npy'), allow_pickle=True)
    targets = np.load(os.path.join('results', params['dataset'], 'targets.npy'), allow_pickle=True)
    if which_expr == 'relpred' or which_expr == 'all': 
        res = evaluate_relationship_prediction(predictions, targets)
    return res

if __name__ == '__main__':
    from settings import params, fix_settings
    fix_settings()
    main(params, sys.argv[1])

