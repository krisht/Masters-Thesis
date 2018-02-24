import numpy as np
import cv2
from matplotlib import pyplot as plt

cap1 = cv2.VideoCapture(1)
cap1.set(3, 160)
cap1.set(4, 120)
cap2 = cv2.VideoCapture(2)
cap2.set(3, 160)
cap2.set(4, 120)

while(True):
	ret, frame = cap1.read()
	ret, frame2 = cap2.read()
	#cv2.imshow('frame', frame)
	#cv2.imshow('frame2', frame2)
	imgL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('gray L', imgL)
	#cv2.imshow('gray R', imgR)

	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	disparity = stereo.compute(imgL,imgR)
	plt.imshow(disparity,'gray')
	plt.show()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
