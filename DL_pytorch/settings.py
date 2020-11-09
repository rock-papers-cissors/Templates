params = {

    'batch_size': 32, 
    # optimizer settings
    'lr': 1e-3,
    # data settings
    'root_dir': '~/Templates/DL_pytorch',
    
}

def fix_settings(args=None):
    if args is not None:
        for k, v in args.items():
            params[k] = v
