import glob
import sys
import os
import scipy.misc
import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Web source: http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
# Source of the code is based on an excelent piece code from stackoverflow
# http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noise_generator (noise_type,img):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= img.shape
    image = img.copy()
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "sp":
        s_vs_p = 0.5
        amount = 0.05 #0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image

def test():
    im = np.asarray(imread('dog.jpg'))

    plt.figure(2)
    sp_im = noise_generator('sp', im)
    gauss_im = noise_generator('gauss', im)
    plt.subplot(1,2,1)
    plt.title('Salt & Pepper Noise')
    plt.imshow(sp_im)
    # plt.title('Original Picture')
    # plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(gauss_im)
    plt.title('Gaussian Noise')
    plt.show()
    plt.close(2)

def createNoise(fileName, newDirName, noiseType):
    im = np.asarray(imread(fileName))
    new_im = noise_generator(noiseType, im)
    print(newDirName + os.path.basename(fileName))
    scipy.misc.imsave(newDirName + os.path.basename(fileName), new_im)

def main(argv):
    dirName = str(argv[0])
    noiseType = str(argv[1])
    filesInDir = glob.glob('./' + dirName + '/*')
    print(filesInDir)

    newDirName = 'NOISE_' + dirName[:-1] + '/'
    if os.path.exists(newDirName):
        print('Directory named ' + newDirName + ' already exists, aborting.')
        return -1
    # os.makedirs(newDirName, 777)
    try:
        original_umask = os.umask(0)
        os.makedirs(newDirName, 777)
    finally:
        os.umask(original_umask)

    for fileName in filesInDir:
        createNoise(fileName, newDirName, noiseType)


if __name__ == "__main__":
    main(sys.argv[1:])