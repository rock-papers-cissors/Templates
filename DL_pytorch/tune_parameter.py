import sys
import settings
import train
import test
import quantitive_evaluate
import itertools
import pandas as pd

def manul_tune():
    args = {
            'key1': [1, 2, 3],
            'key2': ['a', 'b', 'c']
            }
    npdtypes = ['i2', 'a4']
    keys = []
    values = []
    for k, v in args.items():
        keys.append(k)
        values.append(v)
    record_list = []
    for c in itertools.product(*values):
        settings.fix_settings(dict(zip(keys, c)))
        train.main(settings.params)
        test.main(settings.params)
        res = evaluate()
        record_list.append(tuple([*c, res]))
    results = pd.PandasFrame(np.array(
        record_list,
        dtype=[(keys[i], ndtypes[i] for i in range(len(keys)))]
        ))
    results.to_csv()

if __name__ == '__main__':
    
    if sys.args[1] == 'manual':
        manual_tune(args)
