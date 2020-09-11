import cv2 as cv 
import numpy as np
import scipy
import math
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
import sys

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

def get_faceboxes(keypoints_dir, frames_dir, save_dir, phase, start, end, step, myshape, SIZE, boxbuffer, debug=False):

	# myshape = (1080, 1920, 3)
	numframesmade = 0

	boxbuffer = 70

	tary = SIZE
	tarx = 2 * SIZE

	poselen = 69

	saveim = False

	startx = 0
	endx = myshape[1]
	starty = 0
	endy = myshape[0]

	scaley = float(tary) / float(endy - starty)
	scalex = float(tarx) / float(endx - startx)

	if not os.path.exists(save_dir + '/saved_ims/'):
		os.makedirs(save_dir + '/saved_ims/')

	if debug:
		if not os.path.exists(save_dir + '/'+ phase + '_facetexts128/'):
			os.makedirs(save_dir + '/' + phase + '_facetexts128/')

	n = start
	while n <= end:
		framesmadestr = '%06d' % numframesmade
		string_num = '%06d' % n
		key_name = keypoints_dir + "/frame" + string_num

		posepts = readkeypointsfile(key_name + "_pose.yml")

		if len(posepts) != poselen:
			n += 1
			continue
		else:

			print("Getting face bounding box for " + key_name)

			ave = aveface_frompose23(posepts)

			avex = ave[0]
			avey = ave[1]

			minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
			miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
			maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
			maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

			miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx)

			myfile = save_dir + "/"+phase+"_facetexts128/frame" + string_num + '.txt'
			F = open(myfile, "w")
			F.write(str(miny) + " " + str(maxy) + " " + str(minx) + " " + str(maxx))
			F.close()

			if debug:
				frame_name = frames_dir + '/frame' + string_num + ".png"
				if not os.path.isfile(frame_name):
					print('no such frame' + frame_name)
					sys.exit(0)
				else:
					oriImg = cv.imread(frame_name)
					bod = Image.fromarray(oriImg)
					bod.save(save_dir + '/saved_ims/' + 'frame_fUllbod' + string_num + '.png')
					oriImg = Image.fromarray(oriImg[starty:endy, startx:endx, :])
					oriImg = oriImg.resize((2*SIZE,SIZE), Image.ANTIALIAS)
					oriImg = np.array(oriImg)
					oriImg = oriImg[miny:maxy, minx:maxx, [2,1,0]]
					oriImg = Image.fromarray(oriImg)
					oriImg.save(save_dir + '/saved_ims/' + 'frame' + string_num + '.png')

					numframesmade += 1
			n += step
