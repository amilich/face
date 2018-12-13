import sys
import glob
import os

from shutil import copyfile
from matplotlib.image import imread

"""
Go through each folder of images and, for individuals with MIN_PHOTOS or more
pictures, copy NUM_PICTURES to a new output directory.
"""

MIN_PHOTOS = 20
NUM_PICTURES = 20

def main(argv):
    dir_1 = str(argv[0])
    newDirName = str(argv[1])
    print(dir_1)

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
        newSubDir = newDirName + '/' + folderName + '/'
        print('new dir {}'.format(newSubDir))
        if not os.path.exists(newSubDir):
            os.makedirs(newSubDir)
            os.chmod(newSubDir, 0o777)
        # count number of items in each dir
        subdir_name = dir_1 + '/' + folderName
        num_files = len([name for name in os.listdir(subdir_name)\
                        if os.path.isfile(os.path.join(subdir_name, name))])
        print('There are {} files in {}'.format(num_files, subdir_name))
        if num_files < MIN_PHOTOS:
            continue
        count = 0
        for filename in os.listdir(subdir_name):
            img_file = dir_1 + '/' + folderName + '/' + filename 
            dest_1 = newSubDir + '/' + filename
            copyfile(img_file, dest_1)
            count += 1
            if count >= NUM_PICTURES:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
