import sys
sys.path.append("src")

import argparse
import numpy as np
from viz import CallbackContainer

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()

    aparser.add_argument('--input')
    
    args = aparser.parse_args()

    callback = CallbackContainer.load(args.input)

    callback.visualize_mesh()
