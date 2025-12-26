import numpy as np
from config import *
from utils import *
from cross_validation import *
#from intratest_validation import *

'''
Please note that if you run python main.py, it use the default setting  
python main.py --model MDNet --multi-loss 1 --fold 32 --lr 1e-3 --max-eopch 100 --test LOSO-CV -sub-id 0
'''

if __name__ == '__main__':
    args = set_config()
    seed_all(args.random_seed)

    cv = CrossValidation(args)
    cv.n_fold_CV(reproduce=args.reproduce)
