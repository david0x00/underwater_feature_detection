import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
frame_number = 480
frame_difference = 30
filename = "pool_constant_camera.mp4"
# filename = "UnderPool.mp4"

frame_number = 480
frame_difference = 20
filename = "Cycling.mp4"

# frame_number = 200
# frame_difference = 30
# filename = "Coral.mp4"

all_frames = []
def showFrameNum(num=400):
    if all_frames != []:
        return all_frames[num]
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    while success:
        all_frames.append(image)
        success,image = vidcap.read()
        count += 1
    #print('worked')
    # plt.imshow(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
    #plt.imshow(best_image)
    return all_frames[num]

def save_frame(img, name):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)

    output=cv2.drawKeypoints(gray,kp, None)

    filename = 'extracted_frame' + name +'.jpg'
    cv2.imwrite(filename, img)
    print('wrote ' + filename)

img1 = showFrameNum(frame_number)
img2 = showFrameNum(frame_number + frame_difference)
print(img1.shape)
print(img2.shape)

# maxsize = min(img1.shape[0], img1.shape[1])
# img1 = img1[:maxsize, maxsize//2 : maxsize + maxsize//2]
# img2 = img2[:maxsize, maxsize//2 : maxsize + maxsize//2]

save_frame(img1,"1")
save_frame(img2,"2")

vid1_imgs = []
vid2_imgs = []
smoothing_factor = 10
for i in range(frame_number, frame_number + smoothing_factor):
    vid1_imgs.append(showFrameNum(i))
    vid2_imgs.append(showFrameNum(i+frame_difference))

import numpy as np
vid1_imgs = np.array(vid1_imgs)
vid2_imgs = np.array(vid2_imgs)

img1 = np.amax(vid1_imgs, axis=0)
img2 = np.amax(vid2_imgs, axis=0)

save_frame(img1,"1_max")
save_frame(img2,"2_max")


img1 = np.mean(vid1_imgs, axis=0).astype('uint8')
img2 = np.mean(vid2_imgs, axis=0).astype('uint8')

save_frame(img1,"1_mean")
save_frame(img2,"2_mean")
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