import sys
import glob
import os
import scipy.misc
import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.image import imread

def main(argv):
    dir_1 = str(argv[0])
    dir_2 = str(argv[1])
    print(dir_1, dir_2)

    newDirName = 'NOISE_merged/'
    if os.path.exists(newDirName):
        print('Directory named ' + newDirName + ' already exists')
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
        newSubDir = newDirName + '/' + 'NOISE_' + folderName + '/'
        print('new dir {}'.format(newSubDir))
        if not os.path.exists(newSubDir):
            os.makedirs(newSubDir)
            os.chmod(newSubDir, 0o777)
        for filename in os.listdir(dirName + '/' + folderName):
            print(filename)
            # copyfile(filename, dst)

if __name__ == "__main__":
    main(sys.argv[1:])