import argparse
import logging
import os
import pickle
import sys
import time
import hashlib

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from lfw_input import filter_dataset, split_dataset, get_dataset, get_hash_idx
from facenet_external import lfw_input

def main():
	model_exp = os.path.expanduser('/face/etc/20170511-185253/20170511-185253.pb')
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

if __name__ == '__main__':
	main()