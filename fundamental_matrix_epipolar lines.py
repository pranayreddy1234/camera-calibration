###############
##1. Design the function "rectify" to  return
# fundamentalMat: should be 3x3 numpy array to indicate fundamental matrix of two image coordinates. 
# Please check your fundamental matrix using "checkFunMat". The mean error should be less than 5e-4 to get full point.
##2. Design the function "draw_epilines" to  return
# draw1: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the left image; 
# draw2: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the right image.
# See the example of epilines on the PDF.
###############
from cv2 import imread, xfeatures2d, FlannBasedMatcher, cvtColor, COLOR_RGB2BGR, line, circle, computeCorrespondEpilines
import numpy as np
from matplotlib import pyplot as plt

def rectify(pts1, pts2):
    #...
    left_image = np.array(pts1)
    right_image = np.array(pts2)
    xLxR_yLyR = np.multiply(left_image,right_image)
    xRyL = np.multiply(right_image[:,0],left_image[:,1])
    xLyR = np.multiply(left_image[:,0],right_image[:,1])
    A = np.array([xLxR_yLyR[:,0],xRyL,right_image[:,0],xLyR,xLxR_yLyR[:,1],
              right_image[:,1],left_image[:,0],left_image[:,1],np.ones(61,)]).T
    ATA = np.dot(A.T,A)
    U,S,V = np.linalg.svd(ATA, full_matrices=True, compute_uv=True)
    F = U[:,8]
    F = F.reshape(3,3).T
    return F



def draw_epilines(img1, img2, pt1, pt2, fmat):
    #...
    left_line = computeCorrespondEpilines(np.array(pt1).reshape(-1,1,2), 2,fmat)
    left_line = left_line.reshape(3,-1)
    right_line = computeCorrespondEpilines(np.array(pt2).reshape(-1,1,2), 1,fmat)
    right_line = right_line.reshape(3,-1)
    r1,k1,r2,k2=img1.shape[0],img1.shape[1],img2.shape[0],img2.shape[1]
    val1,val2 =[0, int(-1*right_line[2]/right_line[1])]
    val3,val4 =[k1,int(-(right_line[2]+right_line[0]*k1)/right_line[1])]
    fig1=line(img1,(val1,val2),(val3,val4),color=(0,0,0),thickness=4)
    fig1=circle(img1,(int(pt1[0]),int(pt1[1])),radius=10,color=(200,50,40),thickness=-5)
    val5,val6=[0,int(-left_line[2]/left_line[1])]
    val7,val8=[k2,int(-(left_line[2]+left_line[0]*k1)/left_line[1])]
    fig2=line(img2,(val5,val6),(val7,val8),color=(0,0,0),thickness=4)
    fig2=circle(img2,(int(pt2[0]),int(pt2[1])),radius=10,color=(200,50,40),thickness=-5)
    return fig1,fig2



def checkFunMat(pts1, pts2, fundMat):
    N = len(pts1)
    assert len(pts1)==len(pts2)
    errors = []
    for n in range(N):
        v1 = np.array([[pts1[n][0], pts1[n][1], 1]])#size(1,3)
        v2 = np.array([[pts2[n][0]], [pts2[n][1]], [1]])#size(3,1)
        error = np.abs((v1@fundMat@v2)[0][0])
        errors.append(error)
    error = sum(errors)/len(errors)
    return error
    
if __name__ == "__main__":
    img1 = imread('rect_left.jpeg') 
    img2 = imread('rect_right.jpeg')

    # find the keypoints and descriptors with SIFT
    sift = xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters for points match
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    dis_ratio = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            good.append(m)
            dis_ratio.append(m.distance/n.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    min_idx = np.argmin(dis_ratio) 
    
    # calculate fundamental matrix and check error
    fundMat = rectify(pts1, pts2)
    error = checkFunMat(pts1, pts2, fundMat)
    print(error)
    
    # draw epipolar lines
    draw1, draw2 = draw_epilines(img1, img2, pts1[min_idx], pts2[min_idx], fundMat)
    
    # save images
    fig, ax = plt.subplots(1,2,dpi=200)
    ax=ax.flat
    ax[0].imshow(draw1)
    ax[1].imshow(draw2)
    fig.savefig('rect.png')