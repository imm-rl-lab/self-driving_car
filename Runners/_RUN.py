import os, sys
sys.path.insert(0, os.path.abspath('..'))
from Utilities.UniversalRunner import UniversalRunner

if __name__ == '__main__':
    UniversalRunner('ArrivalCarCircle_with_CCEM', {'env_name': ['ArrivalCar'], 'dt': [0.1], 'p': [80], 'lr':[1e-2], 'tau':[1e-2], 'bs': [64], 'lrpf':[16], 'en': [5]}, parallel=False, attempt_n=1, with_seeds=True)
