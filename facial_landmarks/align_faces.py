# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'

def align_face(path, output_path, width):
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
	fa = FaceAligner(predictor, desiredFaceWidth=width)

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(path)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	cv2.imshow("Input", image)
	rects = detector(gray, 2)

	rect = rects[0]
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=width)
	faceAligned = fa.align(image, gray, rect)

	import uuid
	# f = 'resize' # str(uuid.uuid4())
	# print("output/" + f + ".jpg")
	print('writing to {}'.format(output_path))
	cv2.imwrite(output_path, faceAligned)

# align_face('input/gw1.jpg')

# display the output images
# cv2.imshow("Original", faceOrig)
	# cv2.imshow("Aligned", faceAligned)
	# cv2.waitKey(0)