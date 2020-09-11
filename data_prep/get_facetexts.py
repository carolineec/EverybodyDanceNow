import cv2 as cv 
import numpy as np
import scipy
import math
import time
import copy
# import matplotlib
# #%matplotlib inline
# import pylab as plt
# import json
from PIL import Image
from shutil import copyfile
from skimage import img_as_float
from functools import reduce
from renderopenpose import *
import os

myshape = (1080, 1920, 3)

disp = False

start = 115
end = 1999
step = 1
numframesmade = 0
n = start
SIZE = 512
boxbuffer = 70

tary = 512
tarx = 512

"""
	minx: 37.3697357, maxx: 1807.698, miny: 8.71109581, maxy: 1044.72803
	corysshort 822-129502
	coryllong 0 - 22000
	juju 228 - 2298
	miko 600 - 4499
	bruno mars 23 - 4960
	better bruno mars - 0 4982
	Misty (30fps) - (1384, 1540)
"""

poselen = 69

saveim = True

# startx = 300
# endx = 1500
# starty = 20
# endy = 990
startx = 0
endx = 512
starty = 0
endy = 512

scaley = float(tary) / float(endy - starty)
scalex = float(tarx) / float(endx - startx)

def makebox128(miny, maxy, minx, maxx, dimy=128, dimx=128):
	diffy = maxy - miny
	diffx = maxx - minx
	# print "diffyb", maxy - miny
	# print "diffxb", maxx - minx
	if diffy != dimy:
		howmuch = dimy - diffy

		maxy = maxy + (howmuch //2)
		miny = maxy - dimy

		if maxy > 512:
			maxy = 512
			miny = 512 - dimy
		roomtoedge = miny
		if miny < 0:
			miny = 0
			maxy = dimy
	if diffx != dimx:
		howmuch = dimx - diffx

		maxx = maxx + (howmuch //2)
		minx = maxx - dimx

		if maxx > 1024:
			maxx = 1024
			minx = 1024 - dimx
		roomtoedge = minx
		if minx < 0:
			minx = 0
			maxx = dimx

	# print "diffy", maxy - miny
	# print "diffx", maxx - minx
	return miny, maxy, minx, maxx


while n <= end:
	print numframesmade, n
	framesmadestr = '%06d' % numframesmade
	string_num = '%06d' % n
	key_name = "/media/hdd5tb/caroline/keypoints/lingjie/frame" + string_num
	# framenum = '%06d' % n
	# frame_name = "frames/wholedance/frame" + string_num + ".png"

	posepts25 = readkeypointsfile(key_name + "_pose.yml")
	posepts = map_25_to_23(posepts25)
	# oriImg = cv.imread(frame_name)

	if len(posepts) != poselen:
		# print "more than 1 body or read file wrong, " + str(len(posepts))
		# print "EMPTY"
		n += 1
		continue
	else:

		ave = aveface(posepts)

		avex = ave[0]
		avey = ave[1]

		minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
		miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
		maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
		maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

		# print miny,maxy, minx,maxx

		if maxx >= 512:
			print "BEEP" + str(n)
		if maxy >= 512:
			print "ALSDFKJKJ" + str(n)

		miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx)

		print ave, miny, maxy, minx, maxx, string_num

		myfile = "/media/hdd5tb/caroline/train_facetexts128/frame" + string_num + '.txt'
		F = open(myfile, "w")
		print(myfile)
		F.write(str(miny) + " " + str(maxy) + " " + str(minx) + " " + str(maxx))
		F.close()

		if saveim:
			frame_name = "/media/hdd5tb/caroline/lingjie/train_img/frame" + string_num + ".png"
			if not os.path.isfile(frame_name):
				print('bad', frame_name)
			else:
				#"frames/wholedance/frame" + string_num + ".png"
				oriImg = cv.imread(frame_name)
				# oriImg = oriImg[starty:endy, startx:endx, :]
				oriImg = Image.fromarray(oriImg[starty:endy, startx:endx, :])
				oriImg = oriImg.resize((512,512), Image.ANTIALIAS)
				oriImg = np.array(oriImg)
				oriImg = oriImg[miny:maxy, minx:maxx, [2,1,0]]
				oriImg = Image.fromarray(oriImg)
				oriImg.save('/media/hdd5tb/caroline/lingjie/saved_ims/' + 'frame' + string_num + '.png')

				# frame_name2 = '/home/eecs/cchan14/OUTS/val/vcl2_please_full/test_latest/images/frame' + string_num + "_synthesized_image.png"
				# if not os.path.isfile(frame_name2):
				# 	print('bad', frame_name2)
				# #"frames/wholedance/frame" + string_num + ".png"
				# oriImg = cv.imread(frame_name2)
				# # oriImg = oriImg[starty:endy, startx:endx, :]
				# oriImg = Image.fromarray(oriImg[starty:endy, startx:endx, :])
				# oriImg = oriImg.resize((1024,512), Image.ANTIALIAS)
				# oriImg = np.array(oriImg)
				# oriImg = oriImg[miny:maxy, minx:maxx, [2,1,0]]
				# oriImg = Image.fromarray(oriImg)
				# oriImg.save('vcl_faceboxes_txt/saved_ims/noface/' + 'frame' + string_num + '.png')

				numframesmade += 1
		n += step

