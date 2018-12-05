# from facial landmarks directory run:
# python landmarks_nn.py --input-dir=input --output-dir=output

import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import tensorflow as tf
import csv
import cv2
import numpy as np
import scipy.misc

import matplotlib
matplotlib.use("TkAgg")

from tensorflow import keras as keras
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib import cm
from PIL import Image

from align_dlib import *

try:
    from facenet_external.align_dlib import AlignDlib
except:
    from align_dlib import *

from align_faces import align_face

# Usage
# python landmarks_nn.py --input-dir=input --output-dir=output

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def preprocess_image(input_path, output_path, crop_dim, model):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    temp_output = 'tmp/resize_1.jpg'
    align_face(input_path, temp_output, 96)
    x_img = cv2.imread(temp_output)
    x_flat = np.zeros((96,96,1))
    x_flat[:,:,0] = x_img[:,:,0]
    Y = model.predict_on_batch(np.array([x_flat]))[0]
    print(Y)
    x_unflat = np.copy(x_img)
    for i in range(0,Y.shape[0],2):
        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:
            x,y = int(Y[i+1]),int(Y[i])
            # TODO noise around x,y
            x_unflat[x,y,0] = 0
            x_unflat[x,y,1] = 0
            x_unflat[x,y,2] = 0
    cv2.imwrite(temp_output, x_unflat)

    image = _process_image(temp_output, crop_dim)
    if image is not None:
        logger.warning('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, image)
    else:
        logger.warning("Skipping filename: {}".format(input_path))


def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


def load_dataset():
    '''
    Load training dataset
    '''
    Xtrain = []
    Ytrain = []
    with open('input/training/training.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,1), dtype=np.float)
            for i, val in enumerate(row["Image"].split(" ")):
                img[i//IMAGE_WIDTH,i%IMAGE_WIDTH,0] = val
            Yitem = []
            failed = False
            for coord in row:
                if coord == "Image":
                    continue
                if(row[coord].strip()==""):
                    failed = True
                    break
                Yitem.append(float(row[coord]))
            if not failed:
                Xtrain.append(img)
                Ytrain.append(Yitem)
                
    return np.array(Xtrain), np.array(Ytrain, dtype=np.float)


def show_image(X, Y):
    img = np.copy(X)
    # print('hi')
    # plt.imshow(X[:,:,0])
    # plt.show()
    for i in range(0,Y.shape[0],2):
        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:
            x,y = int(Y[i+1]),int(Y[i])
            img[x,y,0] = 0
        else:
            print('Fail')
    plt.imshow(img[:,:,0])
    plt.show()
    # plt.savefig('example.png')

# Preview dataset samples
# show_image(Xtrain[0], Ytrain[0])

def load_test_ex():
    # resize https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    img = Image.open('resize.png')
    basewidth = 96
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    img.save('resize2.png') 
    A = imread('resize2.png')
    return A

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=160, dest='crop_dim',
                        help='Size to crop images to')
    args = parser.parse_args()

    input_dir, output_dir, crop_dim = args.input_dir, args.output_dir, args.crop_dim

    start_time = time.time()


    create_model = False
    if create_model:
        Xdata, Ydata = load_dataset()
        Xtrain = Xdata[:]
        Ytrain = Ydata[:]
        model = keras.Sequential([keras.layers.Flatten(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,1)),
                                  # keras.layers.Conv1D(1, (6), use_bias=True),
                                  keras.layers.Dense(128, activation="relu"),
                                  keras.layers.Dropout(0.1),
                                  keras.layers.Dense(64, activation="relu"),
                                  keras.layers.Dense(30)
                                 ])
        # Compile model
        model.compile(optimizer=tf.train.AdamOptimizer(), 
                      loss='mse',
                      metrics=['mae'])
        # Train model
        model.fit(Xtrain, Ytrain, epochs=500)
        model_json = model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('model.h5')
        print('Saved model to disk')
    else:
        # x = load_test_ex()
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into model
        model.load_weights('model.h5')

    # pool = mp.Pool(processes=mp.cpu_count())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        print('Processing {}'.format(image_path))
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        # pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, model))
        preprocess_image(image_path, output_path, crop_dim, model)
    # pool.close()
    # pool.join()
    logger.info('Preprocessing completed in {} seconds'.format(time.time() - start_time))


    
    # print('Loaded model from disk')
    # print(x.shape)

    # x_dim = np.zeros((96,96,1))
    # x_dim[:,:,0] = x[:,:,0]

    # pred = model.predict_on_batch(np.array([x_dim]))
    # print(pred)
    # show_image(x_dim, pred[0])
        # plt.imshow(x_dim[:,:,0])
        # plt.show()

# python preprocess.py --input-dir=../facial_landmarks  --output-dir=../facial_landmarks/output/ --crop-dim=96
# python ../facenet_external/preprocess.py --input-dir=../in-images  --output-dir=../facial_landmarks/output/ --crop-dim=160
