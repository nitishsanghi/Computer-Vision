import numpy as np # imports the numpy module which has functions related to manupulating arrays and implementing mathematical functions
import cv2 # imports opencv library for python. Image and video processing functions are available.
import glob # imports file handling library
import matplotlib.image as mpimg # imports library for plotting and displaying mathematical functions
import matplotlib.pyplot as plt # imports library for plotting and displaying mathematical functions
from mpl_toolkits.mplot3d import Axes3D # imports library for plotting 3D plots/graphs
from sklearn.preprocessing import StandardScaler # imports Machine learning library. Specifically normalizing function.
from sklearn.model_selection import train_test_split # imports machine learning library. Specifically data splitting function.
from sklearn.svm import LinearSVC # imports machine learning library. Specifically state vector machine model.
from sklearn import svm # Imports SVM module
from skimage.feature import hog # imports machine learning library. Specifically hog function.
from scipy.ndimage.measurements import label # Labels nonzero values in arrays
from moviepy.editor import VideoFileClip
import pickle
import joblib

class vehicleDetector:

	def __init__(self):
		self.svc = None
		self.framesheat = []
		self.scaled_X = None
	# Image Color Histograms. 
	def color_hist(self, img, nbins=32, bins_range=(0, 256)):
		ch1hist = np.histogram(img[:,:,0],nbins, bins_range)
		ch2hist = np.histogram(img[:,:,1],nbins, bins_range)
		ch3hist = np.histogram(img[:,:,2],nbins, bins_range)
		hist_features = np.concatenate((ch1hist[0], ch2hist[0], ch3hist[0]))
		return hist_features

	# HOG refers to histogram of gradients not Ground Hog. 
	def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
		# Call with two outputs if vis==True
		if vis == True:
			features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), 
									transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
			return features, hog_image
		# Otherwise call with one output
		else:      
			features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), 
							transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
			return features

	# Spatial Binning. 
	def bin_spatial(self, img, size=(32, 32)):
		features = cv2.resize(img, size).ravel() 
		return features

	# Convert image to color space
	def convert_color(self, img, conv='RGB2YCrCb'): 
		colorconvDict = {'RGB2YCrCb': cv2.COLOR_RGB2YCrCb,'BGR2YCrCb': cv2.COLOR_BGR2YCrCb, 'RGB2LUV': cv2.COLOR_RGB2LUV, 
						'RGB2HSV': cv2.COLOR_RGB2HSV, 'RGB2HLS': cv2.COLOR_RGB2HLS, 'RGB2YUV': cv2.COLOR_RGB2YUV}
		return cv2.cvtColor(img, colorconvDict[conv])

	# Feature extraction. 
	def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
		features = []
		colorconvDict = {'YCrCb': cv2.COLOR_RGB2YCrCb,'LUV': cv2.COLOR_RGB2LUV, 'HSV': cv2.COLOR_RGB2HSV, 'HLS': cv2.COLOR_RGB2HLS, 'YUV': cv2.COLOR_RGB2YUV}
		for file in imgs:
			file_features = []
			image = mpimg.imread(file)
			if color_space != 'RGB': # Checks and converts to the specified color space
				feature_image = cv2.cvtColor(image, colorconvDict[color_space])

			else: feature_image = np.copy(image)      

			if spatial_feat == True:
				spatial_features = self.bin_spatial(feature_image, size=spatial_size)
				# 3072
				file_features.append(spatial_features) 
			if hist_feat == True:
				hist_features = self.color_hist(feature_image, nbins=hist_bins)
				file_features.append(hist_features)
			if hog_feat == True:
				if hog_channel == 'ALL':
					hog_features = []
					for channel in range(feature_image.shape[2]):
						hog_features.append(self.get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
					hog_features = np.ravel(hog_features)        
				else:
					hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
				file_features.append(hog_features)
			features.append(np.concatenate(file_features))
		return features
		
	# Normalizer: The function takes in the feature list and scales and normalizes it to prevent dominance of a certain
	# feature on the training. The features are also centered for zero mean.
	def normscale(self, feature_list):
		X = np.vstack(feature_list).astype(np.float64)
		X_scaler = StandardScaler().fit(X)
		scaled_X = X_scaler.transform(X)
		return (scaled_X,X_scaler)

	# State Vector Machine Classifier
	def svmclass(self, car_features, noncar_features):
		self.scaled_X = detector.normscale([car_features,noncar_features])
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(self.scaled_X[0], y, test_size=0.3, random_state=rand_state)
		svc = LinearSVC()
		svc.fit(X_train,y_train)
		"""with open('model_pickle.pkl','wb') as f:
    		pickle.dump(svc, f)

		joblib.dump(svc, "model_joblib.pkl")"""
		return svc

	# Collection all vehicle and non-vehicle images together: The function accumulates the filenames of the image data.
	def imgpathaccu(self, foldername):
		print("Reading images from folder")
		imagepathlist = []
		for folder in glob.glob(foldername+'/*'):
			for image in glob.glob(folder+'/*'):
					imagepathlist.append(image)
		return imagepathlist


	# Slide window function creates a list of coordinate points where the windows will be 'drawn' on the image to collect 
	# image data for detecting an tracking vehicles.
	def slide_window(self, img, x_start_stop = [None, None], y_start_stop = [None, None], xy_window = (64, 64), xy_overlap = (0.5, 0.5)):
		# If x and/or y start/stop positions not defined, set to image size
		if x_start_stop[0] == None:
			x_start_stop[0] = 0
		if x_start_stop[1] == None:
			x_start_stop[1] = img.shape[1]
		if y_start_stop[0] == None:
			y_start_stop[0] = 0
		if y_start_stop[1] == None:
			y_start_stop[1] = img.shape[0]
		# Compute the span of the region to be searched    
		xspan = x_start_stop[1] - x_start_stop[0]
		yspan = y_start_stop[1] - y_start_stop[0]
		# Compute the number of pixels per step in x/y
		nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
		ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
		# Compute the number of windows in x/y
		nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
		ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
		nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
		ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
		# Initialize a list to append window positions to
		window_list = []
		# Loop through finding x and y window positions
		# Note: you could vectorize this step, but in practice
		# you'll be considering windows one by one with your
		# classifier, so looping makes sense
		for ys in range(ny_windows):
			for xs in range(nx_windows):
				# Calculate window position
				startx = xs*nx_pix_per_step + x_start_stop[0]
				endx = startx + xy_window[0]
				starty = ys*ny_pix_per_step + y_start_stop[0]
				endy = starty + xy_window[1]
				
				# Append window position to list
				window_list.append(((startx, starty), (endx, endy)))
		# Return the list of windows
		return window_list

	# Define a function to draw bounding boxes. This should be obvious. 
	def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
		# Make a copy of the image
		imcopy = np.copy(img)
		# Iterate through the bounding boxes
		for bbox in bboxes:
			# Draw a rectangle given bbox coordinates
			cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
		# Return the image copy with boxes drawn
		return imcopy

	def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):    
		#1) Define an empty list to receive features
		img_features = []
		#2) Apply color conversion if other than 'RGB'
		colorconvDict = {'YCrCb': cv2.COLOR_RGB2YCrCb,'LUV': cv2.COLOR_RGB2LUV, 'HSV': cv2.COLOR_RGB2HSV, 'HLS': cv2.COLOR_RGB2HLS, 'YUV': cv2.COLOR_RGB2YUV}
		if color_space != 'RGB':
			feature_image = cv2.cvtColor(img, colorconvDict[color_space])

		else: feature_image = np.copy(img)      
		#Resize image
		feature_image = cv2.resize(feature_image, (64,64))
		
		#3) Compute spatial features if flag is set
		if spatial_feat == True:
			spatial_features = self.bin_spatial(feature_image, size=spatial_size)
			#4) Append features to list
			img_features.append(spatial_features)
		#5) Compute histogram features if flag is set
		if hist_feat == True:
			hist_features = self.color_hist(feature_image, nbins=hist_bins)
			#6) Append features to list
			img_features.append(hist_features)
		#7) Compute HOG features if flag is set
		if hog_feat == True:
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.extend(self.get_hog_features(feature_image[:,:,channel], 
										orient, pix_per_cell, cell_per_block, 
										vis=False, feature_vec=True))      
			else:
				hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
							pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			#8) Append features to list
			img_features.append(hog_features)

		#9) Return concatenated array of features
		return np.concatenate(img_features)

	# Define a function you will pass an image 
	# and the list of windows to be searched (output of slide_windows())
	def search_windows(self, img, windows, clf, scaler, color_space='HSV', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True):

		#1) Create an empty list to receive positive detection windows
		on_windows = []
		#2) Iterate over all windows in the list
		for window in windows:
			#3) Extract the test window from original image
			test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
			#4) Extract features for that window using single_img_features()
			features = self.single_img_features(test_img, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, 
											cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
			#5) Scale extracted features to be fed to classifier
			test_features = scaler.transform(np.array(features).reshape(1, -1))
			#6) Predict using your classifier
			prediction = clf.predict(test_features)
			#7) If positive (prediction == 1) then save the window
			if prediction == 1:
				on_windows.append(window)
		#8) Return windows for positive detections
		return on_windows

	def find_cars(self, img, ystart, ystop, scale, svc, X_scaler ,orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
		
		draw_img = np.copy(img)
		img = img.astype(np.float32)/255
		
		img_tosearch = img[ystart:ystop,:,:]
		ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2HSV')
		if scale != 1:
			imshape = ctrans_tosearch.shape
			ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
			
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]

		# Define blocks and steps as above
		nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
		nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
		nfeat_per_block = orient*cell_per_block**2
		
		# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
		window = 64
		nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
		cells_per_step = 2  # Instead of overlap, define how many cells to step
		nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
		nysteps = (nyblocks - nblocks_per_window) // cells_per_step
		
		# Compute individual channel HOG features for the entire image
		hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
		
		boxlist = []
		for xb in range(nxsteps):
			for yb in range(nysteps):
				ypos = yb*cells_per_step
				xpos = xb*cells_per_step
				# Extract HOG for this patch
				hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

				xleft = xpos*pix_per_cell
				ytop = ypos*pix_per_cell

				# Extract the image patch
				subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
			  
				# Get color features
				spatial_features = self.bin_spatial(subimg, size=(spatial_size,spatial_size))
				hist_features = self.color_hist(subimg, nbins=hist_bins)

				# Scale features and make a prediction
				test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
				#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
				test_prediction = svc.predict(test_features)
				
				if test_prediction == 1:
					xbox_left = np.int(xleft*scale)
					ytop_draw = np.int(ytop*scale)
					win_draw = np.int(window*scale)
					cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
					boxlist.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
		return (draw_img,boxlist)

	def add_heat(self, heatmap, bbox_list):
		if len(bbox_list)>0:
			for box in bbox_list:
					heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		else:
			pass
		return heatmap

	def apply_threshold(self, heatmap, threshold):
		heatmap[heatmap <= threshold] = 0
		return heatmap

	def draw_labeled_bboxes(self, img, labels):
			for car_number in range(1, labels[1]+1):
				nonzero = (labels[0] == car_number).nonzero()
				nonzeroy = np.array(nonzero[0])
				nonzerox = np.array(nonzero[1])
				bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
				cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
			return img

	def vdtf(self, img):
		# Defining multiple scales
		scale = [1, 1.25, 1.75, 2, 2.5, 3, 3.5, 4]
		ystart = 400
		ystop = 680 
		
		orient = 9
		pix_per_cell = 8
		cell_per_block = 2
		spatial_size = 32
		hist_bins = 32
		windows= []
		for x in scale:
			if x == 1 or x == 1.25 :
				ystart = 400
				ystop = 496
			elif x == 1.75 :
				ystart = 400
				ystop = 560
			elif x == 2 or x== 2.5:
				ystart = 350
				ystop = 592
			elif x == 3 or x==3.5:
				ystart = 300
				ystop = 680
			else:
				ystart = 300
				ystop = 680
				
			out_img = self.find_cars(img, ystart, ystop, x, self.svc, self.scaled_X[1],orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
			windows = windows + out_img[1]

		heat = np.zeros_like(img[:,:,0]).astype(np.float)
		heat = self.add_heat(heat,windows)
		self.framesheat.append(heat)
		if len(self.framesheat)<20:
			pass
		else:
			for x in self.framesheat[:len(self.framesheat)-2]:
				heat = x +heat
			heat = heat/(len(self.framesheat)+1)
			del self.framesheat[0]

		heat = self.apply_threshold(heat,1.5)
		heatmap = np.clip(heat, 0, 255)
		labels = label(heatmap)
		draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
		return draw_img
	

	def testwindows(self, img):
		windows32 = self.slide_window(img, x_start_stop=[500, None], y_start_stop=[400, 448], xy_window=(32, 32), xy_overlap=(.5, .5))
		windows64 = self.slide_window(img, x_start_stop=[450, None], y_start_stop=[400, 496], xy_window=(64, 64), xy_overlap=(.5, .5))
		windows96 = self.slide_window(img, x_start_stop=[400, None], y_start_stop=[400, 560], xy_window=(96, 96), xy_overlap=(.5, .5))
		windows128 = self.slide_window(img, x_start_stop=[400, None], y_start_stop=[368, 592], xy_window=(128, 128), xy_overlap=(.75, .75))
		windows256 = self.slide_window(img, x_start_stop=[400, None], y_start_stop=[368, 680], xy_window=(256, 256), xy_overlap=(.75, .75))
		
		windows =  windows64 + windows96 + windows128 + windows256
		
		boxedimg = self.draw_boxes(img, windows32, color=(0, 0, 255), thick=2)
		boxedimg = self.draw_boxes(boxedimg, windows64, color=(0, 255, 0), thick=2)
		boxedimg = self.draw_boxes(boxedimg, windows96, color=(255, 0, 0), thick=2)
		boxedimg = self.draw_boxes(boxedimg, windows128, color=(255, 255, 0), thick=2)
		boxedimg = self.draw_boxes(boxedimg, windows256, color=(0, 255, 255), thick=2)
		
		return boxedimg


if __name__ == "__main__":

	detector = vehicleDetector()
	cars = detector.imgpathaccu('vehicles')
	noncars = detector.imgpathaccu('non-vehicles')

	print("Extracting car features")
	car_features = detector.extract_features(cars, color_space = 'HSV', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, 
								cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True)

	print("Extracting non-car features")
	noncar_features = detector.extract_features(noncars,color_space = 'HSV', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, 
									cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True)

	detector.svc = detector.svmclass(car_features, noncar_features)
	vid_output1 = 'Vehicle_Detection_Tracking.mp4'
	clip2 = VideoFileClip("project_video.mp4")
	vid_clip = clip2.fl_image(detector.vdtf) #NOTE: this function expects color images!!
	vid_clip.write_videofile(vid_output1, audio=False)

