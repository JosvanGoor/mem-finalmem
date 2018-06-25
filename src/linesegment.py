import cv2
import numpy as np
from tqdm import tqdm


def getHistogram(image, exp_text_width, height, width):
	exp_text_width = 15
	hist = []
	line_tuples = []
	line_started = 0
	start_line = 0
	end_line = 0
	for i in range(height):
		sum_pixels = width - (sum(image[i])/255.0)
		hist.append(sum_pixels)
	average = (sum(hist)/len(hist))/2
	for count, j in enumerate(hist):
		if j>=average and start_line==0:
			start_line = count
			line_started = 1
		elif j<=average and line_started:
			end_line = count
			if end_line-start_line>exp_text_width:
				line_tuples.append([start_line,count])
			start_line = 0
			line_started = 0
		elif j<average:
			start_line = 0
			line_started = 0
	return line_tuples

def getSliceHist(image, n_slices, exp_text_width, line_array, threshold, overshoot, PSL_width, height, width):
	is_line = False
	line_tuples = []
	start_line = 0
	end_line = 0

	for x in range(0, n_slices):
		left = PSL_width*x
		right = PSL_width*(x+1)
		for y in range(0,height):
			r = range(left, right)
			sum_pixels = PSL_width - (sum(image[y][r])/255.0)
			if sum_pixels > threshold and sum_pixels < PSL_width * 0.8:
				line_array[y] = line_array[y]+1
	left = n_slices*PSL_width
	right = n_slices*PSL_width+overshoot
	for y in range(0,height):
		r = range(left, right)
		sum_pixels = overshoot - (sum(image[y][r])/255.0)
		if sum_pixels > threshold and sum_pixels < overshoot * 0.8:
			line_array[y]=line_array[y]+1

	for count, i in enumerate(line_array):
		if i>=1 and start_line==0:
			start_line = count
		elif i>2 and start_line!=0:
			is_line = True
		elif i<=1 and is_line:
			is_line = False
			end_line = count
			if end_line-start_line>exp_text_width:
				line_tuples.append([start_line,count])
			start_line = 0
		elif i<1:
			start_line = 0

	return line_tuples


def getSeg(image, top, bot):
	imh = image.shape[0]
	imw = image.shape[1]
	seg = np.zeros([imh,0], dtype=np.uint8)
	count = 0
	for i in range(imw):
		if sum(image[0:imh,i])/255<imh:
			if count > 5:
				seg = np.pad(seg,((0,0),(0,10)),mode='constant')
				segw = seg.shape[1]
				seg[0:imh,segw-10:segw] = np.ones([imh,10], dtype=np.uint8)*255
			seg = np.pad(seg,((0,0),(0,1)),mode='constant')
			segw = seg.shape[1]
			seg[0:imh,segw-1] = image[0:imh,i]
			count = 0
		else:
			count += 1
	return seg

def saveSegments(im, imname, line_tuples, pad, height, width, showseg=False, saveseg=False):
	segments = []
	for j, i in enumerate(line_tuples):
		top = i[0]-pad
		bot = i[1]+pad
		segment = im[top:bot,0:width-1]
		seg = getSeg(segment, top, bot)
		segments.append(seg)
		# cv2.imwrite(str(imname)+'seg'+str(j)+('.png'),seg)
		# to get the whole line, without removing white spaces, uncomment the next line
		if saveseg:
			cv2.imwrite(str(imname)+'seg_'+str(j)+'.png',im[top:bot,0:width-1])
		if showseg:
			cv2.imshow('seg',seg)
			cv2.waitKey(0)
	return segments

def showSegments(im, line_tuples, pad, height, width):
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	for j, i in enumerate(line_tuples):
		top = i[0]-pad
		bot = i[1]+pad
		if j%2 == 0:
			cv2.rectangle(im,(10,top),(width-10,bot),(0,0,255),2)
		else:
			cv2.rectangle(im,(10,top),(width-10,bot),(255,0,0),2)
		cv2.imshow('image', im)
		cv2.waitKey(0)


def segmentLine(image, exp_text_width=20, pad=10, PSL_width=128, threshold=8, showseg=0, useHist=0):
	#image = cv2.imread(imname,0)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
	height = image.shape[0]
	width = image.shape[1]

	overshoot = int(width%PSL_width)
	n_slices = int(width/PSL_width)
	threshold = PSL_width/8
	line_array = [0]*height

	if(useHist):
		line_tuples = getHistogram(image, exp_text_width, height, width)
	else:	
		line_tuples = getSliceHist(image, n_slices, exp_text_width, line_array, threshold, overshoot, PSL_width, height, width)
	segments = saveSegments(image, "imname", line_tuples, pad, height, width, showseg)
	if showseg:
		showSegments(image, line_tuples, pad, height, width)

	return segments



#segmentLine("ffilled.png", showseg=0)