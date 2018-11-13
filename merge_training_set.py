import sys
import glob
import os
import scipy.misc
import numpy as np 
import matplotlib.pyplot as plt

from shutil import copyfile
from matplotlib.image import imread

def main(argv):
    dir_1 = str(argv[0])
    dir_2 = str(argv[1])
    print(dir_1, dir_2)

    newDirName = 'NOISE_merged/'
    if os.path.exists(newDirName):
        print('Directory named ' + newDirName + ' already exists')
    else:
        try:
            original_umask = os.umask(0)
            os.makedirs(newDirName)
            os.chmod(newDirName, 0o777)
        finally:
            os.umask(original_umask)

    for folderName in os.listdir(dir_1):
        if folderName.startswith('.'):
            continue
        print(folderName)
        newSubDir = newDirName + '/' + 'MERGED_' + folderName + '/'
        print('new dir {}'.format(newSubDir))
        if not os.path.exists(newSubDir):
            os.makedirs(newSubDir)
            os.chmod(newSubDir, 0o777)
        for filename in os.listdir(dir_1 + '/' + folderName):
            img_file = dir_1 + '/' + folderName + '/' + filename 
            n_img_file = dir_2 + '/' + 'NOISE_' + folderName + '/' + filename
            dest_1 = newSubDir + '/' + filename
            dest_2 = newSubDir + '/' + + filename.split('.')[0] + '_N.' + filename.split('.')[1]
            print('src={} dest={}'.format(img_file, dest_1))
            print('src={} dest={}'.format(n_img_file, dest_2))
            copyfile(img_file, dest_1)
            copyfile(n_img_file, dest_2)


if __name__ == "__main__":
    main(sys.argv[1:])
