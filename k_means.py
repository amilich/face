import sys
import glob
import os
import collections
import random

import scipy.misc
import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt


def get_random_pixel(input_img, past_centroids):
    while True:
        (m,n,d) = input_img.shape
        r_row = random.randint(0,m)
        r_col = random.randint(0,n)
        pixel_val = list(input_img[r_row, r_col, :])
        print(r_row,r_col)
        print(pixel_val)
        print()
        if pixel_val not in past_centroids:
            return tuple(pixel_val)
        else:
            print('Duplicate centroid found')

def find_closest_centroid(row, col, input_img, centroids):
    dists = []
    for centroid in centroids:
        dists.append(sum([(int(input_img[row,col,idx]) - int(centroid[idx]))**2 for idx in range(3)]))
    return dists.index(min(dists))

def k_means(input_img, k):
    (m,n,d) = input_img.shape
    centroids = []
    for idx in range(k):
        centroids.append(get_random_pixel(input_img, centroids))
    num_iter = 0
    while True:
        centroid_assignments = collections.defaultdict(list)
        print('Iter {}'.format(num_iter))
        print(centroids)
        for r_idx in range(m):
            for c_idx in range(n):
                dists = []
                for cent_idx in range(k):
                    dists.append(sum([(float(input_img[r_idx,c_idx,idx]) - \
                                       int(centroids[cent_idx][idx]))**2 for idx in range(3)]))
                centroid_assignments[dists.index(min(dists))].append((r_idx,c_idx))

        new_centroids = []
        for cent_idx,cent_pts in centroid_assignments.items():
            new_centroid = []
            for dim_idx in range(3):
                new_val = sum([input_img[pt[0],pt[1],dim_idx] for pt in cent_pts]) / len(cent_pts)
                new_centroid.append(new_val)
            new_centroids.append(tuple(new_centroid))

        num_iter += 1
        if num_iter >= 30 and set(new_centroids) != set(centroids):
            break
        centroids = new_centroids

    new_img = np.zeros(input_img.shape, dtype=np.int)
    for cent_idx,pts in centroid_assignments.items():
        centroid = centroids[cent_idx]
        for pt in pts:
            for dim_idx in range(3):
                new_img[pt[0], pt[1], dim_idx] = centroid[dim_idx]
    return new_img

def kmeansImage(fileName, newDirName):
    im = np.asarray(imread(fileName))
    new_im = k_means(im, 20)
    print(newDirName + os.path.basename(fileName))
    scipy.misc.imsave(newDirName + os.path.basename(fileName), new_im)

def main(argv):
    dirName = str(argv[0])
    noiseType = str(argv[1])
    print(dirName, noiseType)
    # filesInDir = glob.glob('./' + dirName + '/*')
    # print(filesInDir)

    newDirName = 'KMEANS_' + dirName[:-1] + '/'
    if os.path.exists(newDirName):
        print('Directory named ' + newDirName + ' already exists, aborting.')
        # return -1
    # os.makedirs(newDirName, 777)
    else:
        try:
            original_umask = os.umask(0)
            os.makedirs(newDirName)
            os.chmod(newDirName, 0o777)
        finally:
            os.umask(original_umask)

    for folderName in os.listdir(dirName):
        if folderName.startswith('.'):
            continue
        print(folderName)
        newSubDir = newDirName + '/' + 'KMEANS_' + folderName + '/'
        print('new dir {}'.format(newSubDir))
        if not os.path.exists(newSubDir):
            os.makedirs(newSubDir)
            os.chmod(newSubDir, 0o777)
        for filename in os.listdir(dirName + '/' + folderName):
            print(filename)
            createNoise(dirName + '/' + folderName + '/' + filename, newSubDir, noiseType)


if __name__ == "__main__":
    main(sys.argv[1:])