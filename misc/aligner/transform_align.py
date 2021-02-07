# version 0.25: added params to _detect()
# version 0.2, based on the guide by Adrian Rosebrock at pyimagesearch.com

# switch your editor to tab_size = 4 spaces, you know python is a strange language

import numpy as np
from PIL import Image
import cv2
import dlib
import time
import math # for atan2

from collections import OrderedDict
#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

(L_EYE_LM68_IDX_S, L_EYE_LM68_IDX_E) = FACIAL_LANDMARKS_68_IDXS["left_eye"] # get lm68 start-end indexes
(R_EYE_LM68_IDX_S, R_EYE_LM68_IDX_E) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


class transform_align(object):
	
	def __init__(self, landmark_predictor_weight_path="data_weights/shape_predictor_68_face_landmarks.dat",
			face_detector_path="data_weights", 
			desiredFaceWidth=224, desiredFaceHeight=None, desiredLeftEye=(0.35, 0.38) ):
		"""
		Args:
			landmark_predictor_weight_path (string) full file name of landmarks weights 
			face_detector_path (string) path to detector weights
			desiredFaceHeight (int): image height
			desiredFaceWidth (int): image width
		"""
		self.desiredFaceWidth = desiredFaceWidth
		if desiredFaceHeight == None:
			desiredFaceHeight = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight
		self.desiredLeftEye = desiredLeftEye
		self.detector  = dlib.get_frontal_face_detector() # HOG + SVM
		self.ResNet10SSDdetector = cv2.dnn.readNetFromCaffe(face_detector_path+'/res10_300x300_ssd_iter_140000.txt', 
			face_detector_path+'/res10_300x300_ssd_iter_140000_fp16.caffemodel') 
		self.predictor = dlib.shape_predictor(landmark_predictor_weight_path)
		self.last_OK_face = None
	
	
	#     Parameters
	#    ----------
	#   img: 2D numpy array
	#         The original image with format of (h, w, c)
	# 
	def __call__(self, img):
		"""
		:param img: PIL): Image, RGB 

		:return: aligned face
		"""
		# fr_height = img.size[0]
		# fr_width = img.size[1]
		#img = np.asarray(img)
		#img_dtype = img.dtype
		#img = img.astype(img_dtype)
		#return Image.fromarray(img)
		#print(type(img))
		cv2img = np.array(img) # PIL -> OpenCV RGB
		faces, gray_img = self._detect(cv2img)
			
		# print('a face detected, running align')
		a_cv2face = self._align(image=cv2img, gray=gray_img, rect=faces[0])
		a_pil_face = Image.fromarray(a_cv2face)
		return a_pil_face

	def _detect(self, img, always_detect=1, detect_prob=0.75): 
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # PIL images are RGB
		faces = self.detector(gray_img,0)  # detect bboxes
		if len(faces) == 0 :
			# proceed with the always-detect-something-even-wrong detector:
			(h, w) = img.shape[:2] # original heigh, wid
			#inputBlob = cv2.dnn.blobFromImage(cv2.resize(cv2img, (300, 300)), 1, (300, 300), (104, 177, 123), False) 
			inputBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1, (300, 300), (104, 177, 123), True)
			self.ResNet10SSDdetector.setInput(inputBlob)
			detections = self.ResNet10SSDdetector.forward()
			prediction_score = detections[0, 0, 0, 2] 
			box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h]) # ??
			(x1, y1, x2, y2) = box.astype("int")
			# For better landmark detection
			y1 = int(y1 * 1.15)
			x2 = int(x2 * 1.05)
			x1 = int(x1 * 0.82)
			dlibrect = dlib.rectangle(x1, y1, x2, y2)
			if always_detect>0 or prediction_score>=detect_prob:
				faces.append( dlibrect )
		return faces,gray_img
		
		# for testing:
	def _get_landmarks(self, gray_img, rect):
		landmarks = self.predictor(gray_img, rect) # exec predictor
		landmarks_np = np.zeros((68, 2), dtype=int)# initialize the list of (x, y)-coordinates
		# loop over all facial landmarks and convert them to tuples of (x, y)-coordinates
		for i in range(0, 68):
			landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
		return landmarks_np
	
	def _align(self, image, gray, rect):
		# convert the landmark (x, y)-coordinates to a NumPy array:
		landmarks = self.predictor(gray, rect) # exec predictor
		landmarks_np = np.zeros((68, 2), dtype=int)# initialize the list of (x, y)-coordinates
		# loop over all facial landmarks and convert them to tuples of (x, y)-coordinates
		for i in range(0, 68):
			landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
			# draw landmarks:
			#cv2.circle(image, (landmarks_np[i][0],landmarks_np[i][1]), 2, (255, 0, 0), -1)
		
		# extract the left and right eye (x, y)-coordinates			
		leftEyePts  = landmarks_np[L_EYE_LM68_IDX_S:L_EYE_LM68_IDX_E] # l-eye index
		rightEyePts = landmarks_np[R_EYE_LM68_IDX_S:R_EYE_LM68_IDX_E] # r-eye index

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0)#.astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0)#.astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the  *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

		# return the aligned face
		return output

	def __Norm(self, img):
		"""
		:param img: PIL): Image 

		:return: Normalized image
		"""
		img = np.asarray(img)
		img_dtype = img.dtype

		power = 6
		extra = 6

		img = img.astype('float32')
		img_power = np.power(img, power)
		rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
		rgb_norm = np.power(np.sum(np.power(rgb_vec, extra)), 1 / extra)
		rgb_vec = rgb_vec / rgb_norm
		rgb_vec = 1 / (rgb_vec * np.sqrt(3))
		img = np.multiply(img, rgb_vec)
		img = img.astype(img_dtype)

		return Image.fromarray(img)

	def __repr__(self):
		return self.__class__.__name__+'()'

