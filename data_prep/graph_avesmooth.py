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
from scipy.misc import imresize
from scipy.misc import imsave
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

##### Must specifcy these parameters
parser.add_argument('--keypoints_dir', type=str, default='keypoints', help='directory where target keypoint files are stored, assumes .yml format for now.')
parser.add_argument('--frames_dir', type=str, default='frames', help='directory where source frames are stored. Assumes .png files for now.')
parser.add_argument('--save_dir', type=str, default='save', help='directory where to save generated files')
parser.add_argument('--spread', nargs='+', type=int, help='range of frames to use for target video plus ***window size = number of frames to average over,\
	e.g. 0 10000 4, ')

#### Optional (have defaults)
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--facetexts', action='store_true', help='use this flag to also save face 128x128 bounding boxes')
parser.add_argument('--boxbuffer', type=int, default=70, help='face bounding box width/height')
parser.add_argument('--num_face_keypoints', type=int, default=8, help='number of face keypoints to plot. Acceptable values are 8, 9, 22, 70. \
	If another value is specified, the default number of points will be plotted.')
parser.add_argument('--output_dim', type=int, default=512, help='default width of output images. Output images will have size output_dim, 2*output_dim')
parser.add_argument('--debug', action='store_true', help='use this flag for debugging')

opt = parser.parse_args()

myshape = (1080, 1920, 3)
disp = False

phase = opt.phase

spread = tuple(opt.spread)
start = spread[0]
end = spread[1]
step = 1
w_size = spread[2]
SIZE = opt.output_dim
numkeypoints = opt.num_face_keypoints
get_facetexts = opt.facetexts
boxbuffer = opt.boxbuffer

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

keypoints_dir = opt.keypoints_dir
frames_dir = opt.frames_dir
savedir = opt.save_dir

if not os.path.exists(savedir):
	os.makedirs(savedir)
if not os.path.exists(savedir + '/' + phase +'_label'):
	os.makedirs(savedir + '/'+phase+'_label')
if not os.path.exists(savedir + '/'+phase+'_img'):
	os.makedirs(savedir + '/'+phase+'_img')
if not os.path.exists(savedir + '/'+phase+'_facetexts128') and get_facetexts:
	os.makedirs(savedir + '/'+phase+'_facetexts128')

if not os.path.exists(savedir + '/debug'):
	os.makedirs(savedir + '/debug')

print('----------------- Loading Frames -----------------')
frames = os.listdir(frames_dir)
print('----------------- All Loaded -----------------')

pose_window = []
face_window = []
rhand_window = []
lhand_window = []

original_queue = []

n = start
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
		print "EMPTY"
		n += 1
		continue
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

	pose_window += [posepts]
	face_window += [facepts]
	rhand_window += [r_handpts]
	lhand_window += [l_handpts]

	original_queue += [oriImg]

	if len(pose_window) >= w_size:
		print("Plotting stick figure for last frame " + filebase_name)
		h_span = w_size // 2

		all_pose = np.array(pose_window)
		all_face = np.array(face_window)
		all_rhand = np.array(rhand_window)
		all_lhand = np.array(lhand_window)

		posedivide = np.prod((all_pose[:, 2::3] > 0).astype('float'), axis=0)
		pose_cons = np.zeros(len(pose_window[0]))
		pose_cons[::3] = posedivide
		pose_cons[1::3] = posedivide
		pose_cons[2::3] = posedivide
		ave_posepts = np.sum(all_pose, axis=0) / float(w_size)
		# print ave_posepts
		# print pose_cons
		ave_posepts = ave_posepts * pose_cons
		# print ave_posepts

		facedivide = np.prod((all_face[:, 2::3] > 0).astype('float'), axis=0)
		face_cons = np.zeros(len(face_window[0]))
		face_cons[::3] = facedivide
		face_cons[1::3] = facedivide
		face_cons[2::3] = facedivide
		ave_facepts = np.sum(all_face, axis=0) / float(w_size)
		ave_facepts = ave_facepts * face_cons

		rhanddivide = np.prod((all_rhand[:, 2::3] > 0).astype('float'), axis=0)
		rhand_cons = np.zeros(len(rhand_window[0]))
		rhand_cons[::3] = rhanddivide
		rhand_cons[1::3] = rhanddivide
		rhand_cons[2::3] = rhanddivide
		ave_rhand = np.sum(all_rhand, axis=0) / float(w_size)
		ave_rhand = ave_rhand * rhand_cons

		lhanddivide = np.prod((all_lhand[:, 2::3] > 0).astype('float'), axis=0)
		lhand_cons = np.zeros(len(lhand_window[0]))
		lhand_cons[::3] = lhanddivide
		lhand_cons[1::3] = lhanddivide
		lhand_cons[2::3] = lhanddivide
		ave_lhand = np.sum(all_lhand, axis=0) / float(w_size)
		ave_lhand = ave_lhand * lhand_cons

		ave = aveface(ave_posepts)

		canvas = renderpose(ave_posepts, 255 * np.ones(myshape, dtype='uint8'))
		canvas = renderface_sparse(ave_facepts, canvas, numkeypoints)
		canvas = renderhand(ave_rhand, canvas)
		canvas = renderhand(ave_lhand, canvas)

		canvas = canvas[starty:endy, startx:endx, [2,1,0]]
		canvas = Image.fromarray(canvas)

		saveoriImg = original_queue[h_span]
		saveoriImg = saveoriImg[starty:endy, startx:endx, [2,1,0]]
		saveoriImg = Image.fromarray(saveoriImg)

		saveoriImg = saveoriImg.resize((2*SIZE,SIZE), Image.ANTIALIAS)
		canvas = canvas.resize((2*SIZE,SIZE), Image.ANTIALIAS)

		saveoriImg.save(savedir + '/' + phase + '_img/frame' + framesmadestr + '.png')
		canvas.save(savedir + '/' + phase + '_label/frame' + framesmadestr + '.png')

		if get_facetexts:
			avex = ave[0]
			avey = ave[1]

			minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
			miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
			maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
			maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

			# print miny,maxy, minx,maxx

			miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx)

			# print miny, maxy, minx, maxx

			myfile = savedir + "/" + phase + "_facetexts128/frame" + framesmadestr + '.txt'
			F = open(myfile, "w")
			F.write(str(miny) + " " + str(maxy) + " " + str(minx) + " " + str(maxx))
			F.close()

			debug = True
			if debug:
				oriImg = np.array(saveoriImg) #already 512x1024
				oriImg = oriImg[miny:maxy, minx:maxx, :]
				oriImg = Image.fromarray(oriImg)
				oriImg.save(savedir + '/debug/' + filebase_name + '.png')


		pose_window = []
		face_window = []
		rhand_window = []
		lhand_window = []
		original_queue = []

		numframesmade += 1
	n += step
