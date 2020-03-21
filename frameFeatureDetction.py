import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual

def showFrameNum(num=(1,400,1)):
    vidcap = cv2.VideoCapture('pool_constant_camera.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        if count == num:
            #cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file   
            #print('Wrote a new frame: ', success)
            best_image = image
            break
        success,image = vidcap.read()
        count += 1
    #print('worked')
    plt.imshow(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
    #plt.imshow(best_image)
    return best_image
best = showFrameNum(200)
# interact(showFrameNum)

# src = cv.imread(cv.samples.findFile(args.input), cv.IMREAD_GRAYSCALE)
img = best
# img = cv2.imread('chessboard.jpg')

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print('before a')
sift = cv2.xfeatures2d.SIFT_create()
print('before b')
kp = sift.detect(gray,None)

print('before c')
img=cv2.drawKeypoints(gray,kp, None)

cv2.imwrite('sift_keypoints.jpg',img)

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