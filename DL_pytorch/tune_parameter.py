import sys
import settings
import train
import test
import quantitive_evaluate
import itertools
import pandas as pd
import numpy as np
import os

def manual_tune(args, res_keys, npdtypes):
    keys = []
    values = []
    for k, v in args.items():
        keys.append(k)
        values.append(v)
    keys += res_keys
    record_list = []
    try:
        for c in itertools.product(*values):
            print('Try this setting : ', dict(zip(keys, c)))
            settings.fix_settings(dict(zip(keys, c)))
            train.main(settings.params)
            test.main(settings.params)
            res = quantitive_evaluate.main(settings.params, 'relpred')
            record_list.append(tuple([*c, *[res[k] for k in res_keys]]))
        results = pd.DataFrame(np.array(
            record_list,
            dtype=[(keys[i], npdtypes[i]) for i in range(len(keys))]
            ))
    finally:
        results.to_csv(os.path.join('expr_log.csv'))

if __name__ == '__main__':
    
    if sys.argv[1] == 'manual':
        args = {
            'arg1': ['a', 'b', 'c'],
            'arg2': [True, False]
            }
        npdtypes = ['a4', 'bool', 'f8', 'f8', 'f8']
        res_keys = ['metric1', 'metric2', 'metric3']
        manual_tune(args, res_keys, npdtypes)
