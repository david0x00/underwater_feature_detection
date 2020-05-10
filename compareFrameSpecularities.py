import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
from collections import defaultdict
import sys
import torch
from torch import nn

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
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
    plt.show()
    

def print_layer(printed, name, x):
      if not printed: 
        # print(name)
        # print(x.shape)
        pass
class SegmentationNet(nn.Module):
    def __init__(self): # feel free to modify input paramters
        super(SegmentationNet, self).__init__()
        self.printed = False
        self.TOTAL_CLASSES = 9

        self.conv1 = nn.Conv2d(3, 64, 3,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(2,2,return_indices=True)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=256)
        self.pool2 = nn.MaxPool2d(2,2,return_indices=True)

        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=512)
        self.pool3 = nn.MaxPool2d(2,2,return_indices=True)

        self.deconv1 = nn.ConvTranspose2d(512,512,3,padding=1)
        self.deconv2 = nn.ConvTranspose2d(512,256,3,padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.unpool1 = nn.MaxUnpool2d(2,2)

        self.deconv3 = nn.ConvTranspose2d(256,256,3,padding=1)
        self.deconv4 = nn.ConvTranspose2d(256,128,3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.unpool2 = nn.MaxUnpool2d(2,2)

        self.deconv5 = nn.ConvTranspose2d(128,64,3,padding=1)
        self.deconv6 = nn.ConvTranspose2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.unpool3 = nn.MaxUnpool2d(2,2)


        self.deconv7 = nn.ConvTranspose2d(64,2,3,padding=1)
        self.bn4 = nn.BatchNorm2d(2)


    def forward(self, x): # feel free to modify input paramters
        printed = self.printed

        x = self.batchnorm1(self.relu(self.conv1(x)))
        print_layer(printed, 'conv1', x)
        # x = self.relu(self.batchnorm2(self.conv2(x)))
        x, i1 = self.pool1(self.batchnorm2(self.relu(self.conv3(self.relu(self.conv2(x))))))
        print_layer(printed, 'pool1', x)
        x, i2 = self.pool2(self.batchnorm3(self.relu(self.conv5(self.relu(self.conv4(x))))))
        print_layer(printed, 'pool2', x)
        x, i3 = self.pool3(self.batchnorm4(self.relu(self.conv7(self.relu(self.conv6(x))))))
        print_layer(printed, 'pool3', x)

        x = self.bn1(self.relu(self.deconv2(self.relu(self.deconv1(self.unpool1(x, i3))))))
        print_layer(printed, 'unpool1', x)

        x = self.bn2(self.relu(self.deconv4(self.relu(self.deconv3(self.unpool2(x, i2))))))
        print_layer(printed, 'unpool2', x)

        x = self.bn3(self.relu(self.deconv6(self.relu(self.deconv5(self.unpool3(x, i1))))))
        print_layer(printed, 'unpool3', x)

        x = self.bn4(self.relu(self.deconv7(x)))

        self.printed = True
        return x

from scipy import ndimage
def remove_specularities(img, frame_num):
    net = SegmentationNet()
    net.load_state_dict(torch.load('SpecularityModelStateDict', map_location=torch.device('cpu')))
    net.eval()
    
    temp = np.array([img/255], dtype=np.float32) 
    tensor = torch.from_numpy(temp).permute(0, 3, 1, 2) # default is torch.Size([224, 288, 3])

    mask = net(tensor)
    mask = mask.detach().cpu().numpy()[0][1]
    threshold = 0.5
    mask = (mask > threshold) * 1

    fig = plt.figure()
    plt.imshow(mask, cmap='gray')
    # plt.show()
    plt.savefig("bluring/mask" + str(frame_num) + ".png")
    
    fig = plt.figure()
    plt.imshow(img)
    # plt.show()
    plt.savefig("bluring/unfiltered" + str(frame_num) + ".png")

    # blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
    # blurred_img = cv2.medianBlur(img, 5)
    blurred_img = ndimage.minimum_filter(img, size=(5, 5, 1))
    blurred_img = cv2.medianBlur(blurred_img, 5)
    out = np.array([[blurred_img[i][j] if mask[i][j] else img[i][j] for j in range(len(img[i]))] for i in range(len(img))])

    plt.imshow(out)
    # plt.show()
    plt.savefig("bluring/filtered-min" + str(frame_num) + ".png")
    return out

fig, ax = None, None
def plot_matches(img1, img2, comparison):
    xyxypairs = get_best_matches(img1, img2, 1, threshold)
    fig, ax = plt.subplots(figsize=(20,10))
    plot_inlier_matches(ax, img1, img2, xyxypairs)
    fig.savefig('bluring/' + comparison + '.jpg', bbox_inches='tight')
    plt.show()
start_frame = 0
frame_difference = 120
threshold = 2000
# video_file_name = "pool_constant_camera.mp4"

# frame_number = 0
# frame_difference = 30
# threshold = 1000
# video_file_name = "Coral.mp4"

# start_frame = 0
# frame_difference = 120
# threshold = 50000
# video_file_name = "Cycling.mp4"

# start_frame = 30
# frame_difference = 10
# threshold = 10000
# video_file_name = "Carla.mp4"


video_base = "videos/"

if len(sys.argv) != 3:
    print('Provide video name and frame number. ')
    exit(1)
video_file_name = sys.argv[1]
frame_num = int(sys.argv[2])

frame1 = get_frame_number(video_base+video_file_name, frame_num)
frame1 = cv2.resize(frame1, (288, 224), interpolation=cv2.INTER_LINEAR)

frame2 = get_frame_number(video_base+video_file_name, start_frame + frame_difference)
frame2 = cv2.resize(frame2, (288, 224), interpolation=cv2.INTER_LINEAR)

aggregation_count = 10
# max1, mean1 = create_save_max_mean_images(video_base+video_file_name, start_frame, aggregation_count)
# max2, mean2 = create_save_max_mean_images(video_base+video_file_name, start_frame + frame_difference, aggregation_count)

plot_matches(frame1, frame2, video_file_name[:-3] + 'regular')
frame1 = remove_specularities(frame1, 1)
frame2 = remove_specularities(frame2, 2)
plot_matches(frame1, frame2, video_file_name[:-3] + 'minBlur')

# plot_matches(max1, max2, video_file_name[:-3] + 'max')
# plot_matches(mean1, mean2, video_file_name[:-3] + 'mean')


