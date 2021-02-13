###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    #......
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = imread(imgname)
    img[:,:1000] = [0,0,0]
    gray = cvtColor(img, COLOR_BGR2GRAY)
    ret, corners = findChessboardCorners(gray, (4,4),None)
    corners = cornerSubPix(gray,corners,(15,15),(-1,-1),criteria)
    corners = corners.reshape(len(corners),2)
    img1 = imread(imgname)
    img1[:,1000:] = [0,0,0]
    gray1 = cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret1, corners1 = findChessboardCorners(gray1, (4,4),None)
    corners1 = cornerSubPix(gray1,corners1,(15,15),(-1,-1),criteria)
    corners1 = corners1.reshape(len(corners1),2)
    IC = np.concatenate((corners1,corners),axis = 0)
    WC = np.array([[0,20,20,1],
                              [0,15,20,1],
                              [0,10,20,1],
                              [0,5,20,1],
                              [0,20,15,1],
                              [0,15,15,1],
                              [0,10,15,1],
                              [0,5,15,1],
                              [0,20,10,1],
                              [0,15,10,1],
                              [0,10,10,1],
                              [0,5,10,1],
                              [0,20,5,1],
                              [0,15,5,1],
                              [0,10,5,1],
                              [0,5,5,1],
                              [20,0,20,1],
                              [20,0,15,1],
                              [20,0,10,1],
                              [20,0,5,1],
                              [15,0,20,1],
                              [15,0,15,1],
                              [15,0,10,1],
                              [15,0,5,1],
                              [10,0,20,1],
                              [10,0,15,1],
                              [10,0,10,1],
                              [10,0,5,1],
                              [5,0,20,1],
                              [5,0,15,1],
                              [5,0,10,1],
                              [5,0,5,1]])
    ones_array = np.ones((1,12))
    for i in range(32):
        sample_array = np.array([[WC[i,0],WC[i,1],WC[i,2],1,0,0,0,0,-(IC[i,0]*WC[i,0]),-(IC[i,0]*WC[i,1]),-(IC[i,0]*WC[i,2]),-(IC[i,0])],[0,0,0,0,WC[i,0],WC[i,1],WC[i,2],1,-(IC[i,1]*WC[i,0]),-(IC[i,1]*WC[i,1]),-(IC[i,1]*WC[i,2]),-(IC[i,1])]])
        ones_array = np.concatenate((ones_array,sample_array), axis=0)
    required_array = ones_array[1:,:]
    U,S,V = np.linalg.svd(required_array)
    X = V[11,:]
    X = X.reshape(3,4)
    lamda = np.sqrt(1/((X[2,0]**2)+(X[2,1]**2)+(X[2,2]**2)))
    m = lamda * X
    ox = (m[2,0]*m[0,0])+(m[2,1]*m[0,1])+(m[2,2]*m[0,2])
    oy = (m[2,0]*m[1,0])+(m[2,1]*m[1,1])+(m[2,2]*m[1,2])
    fx = ((m[0,0]*m[0,0])+(m[0,1]*m[0,1])+(m[0,2]*m[0,2]) - (ox*ox))**(0.5)
    fy = ((m[1,0]*m[1,0]) + (m[1,1]*m[1,1]) + (m[1,2]*m[1,2]) - (oy*oy))**(0.5)
    return [fx,fy,ox,oy], True
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')