import cv2 as cv 
import numpy as np
import scipy
import math
import time
import copy
import matplotlib
#%matplotlib inline
import pylab as plt
import json
from PIL import Image
from shutil import copyfile
from skimage import img_as_float
from functools import reduce
from renderopenpose import *
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

##### Must specifcy these parameters
parser.add_argument('--keypoints_dir', type=str, default='keypoints', help='directory where target keypoint files are stored, assumes .yml format for now.')
parser.add_argument('--frames_dir', type=str, default='frames', help='directory where source frames are stored. Assumes .png files for now.')
parser.add_argument('--save_dir', type=str, default='save', help='directory where to save generated files')
parser.add_argument('--spread', nargs='+', type=int, help='range of frames to use for target video plus step size [start end step] e.g. 0 10000 1')

#### Optional (have defaults)
parser.add_argument('--facetexts', action='store_true', help='use this flag to also save face 128x128 bounding boxes')
parser.add_argument('--boxbuffer', type=int, default=70, help='face bounding box width/height')
parser.add_argument('--num_face_keypoints', type=int, default=8, help='number of face keypoints to plot. Acceptable values are 8, 9, 22, 70. \
	If another value is specified, the default number of points will be plotted.')
parser.add_argument('--output_dim', type=int, default=512, help='default width of output images. Output images will have size output_dim, 2*output_dim')
parser.add_argument('--map_25_to_23', action='store_true', help='load body keypoints in 25 OpenPose format, but graph in 23 keypoint OpenPose format')
parser.add_argument('--debug', action='store_true', help='use this flag for debugging')

opt = parser.parse_args()

myshape = (1080, 1920, 3)
disp = False

spread = tuple(opt.spread)
start = spread[0]
end = spread[1]
step = spread[2]
SIZE = opt.output_dim #512
numkeypoints = opt.num_face_keypoints
get_factexts = opt.facetexts #True
boxbuffer = opt.boxbuffer #70

numframesmade = 0
n = start

print step

startx = 0
endx = myshape[1]
starty = 0
endy = myshape[0]

tary = SIZE
tarx = 2*SIZE

scaley = float(tary) / float(endy - starty)
scalex = float(tarx) / float(endx - startx)

poselen = [54, 69, 75]

keypoints_dir = opt.keypoints_dir #"/data/scratch/caroline/keypoints/jason_keys"
frames_dir = opt.frames_dir #"/data/scratch/caroline/frames/jason_frames"
savedir = opt.save_dir #"/data/scratch/caroline/omegalul"

if not os.path.exists(savedir):
	os.makedirs(savedir)
if not os.path.exists(savedir + '/train_label'):
	os.makedirs(savedir + '/train_label')
if not os.path.exists(savedir + '/train_img'):
	os.makedirs(savedir + '/train_img')
if not os.path.exists(savedir + '/train_facetexts128'):
	os.makedirs(savedir + '/train_facetexts128')

if opt.debug and (not os.path.exists(savedir + '/debug')):
	os.makedirs(savedir + '/debug')

print('----------------- Loading Frames -----------------')
frames = sorted(os.listdir(frames_dir))
print frames
print('----------------- All Loaded -----------------')

while n <= end:
	print n
	framesmadestr = '%06d' % numframesmade

	filebase_name = os.path.splitext(frames[n])[0]
	key_name = os.path.join(keypoints_dir, filebase_name)
	frame_name = os.path.join(frames_dir, frames[n])

	posepts = []

	### try yaml
	posepts = readkeypointsfile(key_name + "_pose")
	facepts = readkeypointsfile(key_name + "_face")
	r_handpts = readkeypointsfile(key_name + "_hand_right")
	l_handpts = readkeypointsfile(key_name + "_hand_left")
	if posepts is None: ## try json
		posepts, facepts, r_handpts, l_handpts = readkeypointsfile(key_name + "_keypoints")
		if posepts is None:
			print('unable to read keypoints file')
			import sys
			sys.exit(0)

	if not (len(posepts) in poselen):
		# empty or contains multiple detections
		print "empty", len(posepts)
		n += 1
		continue
	else:
		print 'graphing file %s' % filebase_name
		if opt.map_25_to_23:
			posepts = map_25_to_23(posepts)

		oriImg = cv.imread(frame_name)
		curshape = oriImg.shape

		### scale and resize:
		sr = scale_resize(curshape, myshape=(1080, 1920, 3), mean_height=0.0)
		if sr:
			scale = sr[0]
			translate = sr[1]

			oriImg = fix_scale_image(oriImg, scale, translate, myshape)
			posepts = fix_scale_coords(posepts, scale, translate)
			facepts = fix_scale_coords(facepts, scale, translate)
			r_handpts = fix_scale_coords(r_handpts, scale, translate)
			l_handpts = fix_scale_coords(l_handpts, scale, translate)

		canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
		canvas = renderface_sparse(facepts, canvas, numkeypoints, disp=False)
		canvas = renderhand(r_handpts, canvas)
		canvas = renderhand(l_handpts, canvas)

		oriImg = Image.fromarray(oriImg[:, :, [2,1,0]])
		canvas = Image.fromarray(canvas[:, :, [2,1,0]])

		oriImg = oriImg.resize((2*SIZE,SIZE), Image.ANTIALIAS)
		canvas = canvas.resize((2*SIZE,SIZE), Image.ANTIALIAS)

		oriImg.save(savedir + '/train_img/' + filebase_name + '.png')
		canvas.save(savedir + '/train_label/' + filebase_name + '.png')

		""" save factexts """
		if get_factexts:

			ave = aveface(posepts)

			avex = ave[0]
			avey = ave[1]

			minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
			miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
			maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
			maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

			miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx)

			# print miny, maxy, minx, maxx, filebase_name

			myfile = savedir + "/train_facetexts128/" + filebase_name + '.txt'
			F = open(myfile, "w")
			F.write(str(miny) + " " + str(maxy) + " " + str(minx) + " " + str(maxx))
			F.close()

			# debug = True
			if opt.debug:
				oriImg = np.array(oriImg) #already 512x1024
				oriImg = oriImg[miny:maxy, minx:maxx, :]
				oriImg = Image.fromarray(oriImg)
				oriImg.save(savedir + '/debug/' + filebase_name + '.png')

		numframesmade += 1
	n += step
