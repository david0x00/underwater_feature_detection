import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
from collections import defaultdict
import sys

videos = defaultdict(list)
def get_frame_number(video_filename, frame_num):
    frames = videos[video_filename]
    if frames != []:
        return frames[frame_num]
    vidcap = cv2.VideoCapture(video_filename)
    success,image = vidcap.read()
    if not success:
        print('unable to load video `' + video_filename + '`')
        sys.exit(1)
    while success:
        frames.append(image)
        success,image = vidcap.read()

    return frames[frame_num]

def imsave(img, name):
    filepath = './output/' + name
    cv2.imwrite(name, img)
    print('wrote ' + name)

import numpy as np
def create_save_max_mean_images(video_filename, start_frame, aggregate_count):
    vid1_imgs = [] 
    for i in range(start_frame, start_frame + aggregate_count):
        vid1_imgs.append(get_frame_number(video_filename, i))
    vid1_imgs = np.array(vid1_imgs)
    max_img = np.amax(vid1_imgs, axis=0)
    mean_img = np.mean(vid1_imgs, axis=0).astype('uint8')

    base_filepath = './output/' + video_filename[:-3] + '_' + str(start_frame) + '_Agg' + str(aggregate_count)
    # imsave(max_img, base_filepath + '_max.jpg')
    # imsave(mean_img, base_filepath + '_mean.jpg')

    return max_img, mean_img

import scipy
from scipy.spatial import distance
import skimage.io
def get_sift_data(img):
    """
    detect the keypoints and compute their SIFT descriptors with opencv library
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def get_best_matches(img1, img2, num_matches, query_dist=5000):
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    
    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    print(dist.shape)
    idxs1, idxs2 = np.where(dist < query_dist)
    
    bp1 = kp1[idxs1]
    bp2 = kp2[idxs2]
    xyxy = np.array([(bp1[i].pt[0], bp1[i].pt[1], bp2[i].pt[0], bp2[i].pt[1]) for i in range(len(bp2))])
    return xyxy

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    plot the match between two image according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')

start_frame = 0
frame_difference = 30
video_file = "pool_constant_camera.mp4"
# video_file = "UnderPool.mp4"

frame_number = 0
frame_difference = 30
threshold = 1000
video_file = "Coral.mp4"

start_frame = 0
frame_difference = 20
threshold = 10000
video_file = "Cycling.mp4"


video_base = "videos/"
frame1 = get_frame_number(video_base+video_file, start_frame)
frame2 = get_frame_number(video_base+video_file, start_frame + frame_difference)
# imsave(frame1, video_file[:-3] + '.jpg')
# imsave(frame2, video_file[:-3] + '.jpg')

aggregation_count = 10
max1, mean1 = create_save_max_mean_images(video_base+video_file, start_frame, aggregation_count)
max2, mean2 = create_save_max_mean_images(video_base+video_file, start_frame + frame_difference, aggregation_count)

fig, ax = None, None
def plot_matches(img1, img2, comparison):
    xyxypairs = get_best_matches(img1, img2, 1, threshold)
    fig, ax = plt.subplots(figsize=(20,10))
    plot_inlier_matches(ax, img1, img2, xyxypairs)
    fig.savefig('output/' + comparison + '.jpg', bbox_inches='tight')

plot_matches(frame1, frame2, video_file[:-3] + 'regular')
plot_matches(max1, max2, video_file[:-3] + 'max')
plot_matches(mean1, mean2, video_file[:-3] + 'mean')




# def save_frame(img, name):
#     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     sift = cv2.xfeatures2d.SIFT_create()
#     kp = sift.detect(gray,None)

#     output=cv2.drawKeypoints(gray,kp, None)

#     filename = 'extracted_frame' + name +'.jpg'
#     cv2.imwrite(filename, img)
#     print('wrote ' + filename)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
#
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()