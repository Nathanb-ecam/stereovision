"""
1 : calibration de la camera
2 : par cette calibration, on peut récupérer les matrices de rotations et de translations correspondantes pour chaque image 


1 : trouver les matrices projections correspondantes des deux camera P et P' (par calibration de caméra) 
On a mis une ligne rouge sur l'objet avant de prendre des paires de photos (gauche et droite)
2 : on peut determiner la matrice fondamentale a partir à partir de P et P', la matrice fondamentale F permet de trouver la ligne épipolaire (sur l'image de droite) correspondante à un point de l'image de gauche 

Avant de prendre les deux photos des cameras, on trace une ligne rouge sur l'objet qui va permettre de faciliter la correspondances entre un point et sa ligne épipolaire
draw_epilines() dans open cv2
"""


import numpy as np
from numpy.linalg import inv
import cv2 
import glob

def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./chessboards/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (7,7), (-1,-1), criteria)
            # print(corners2)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,7), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            # print(fname)
            
    cv2.destroyAllWindows()
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print("revcs")
    # print(rvecs[0])
 
    return ret,cameraMatrix,dist,rvecs,tvecs
# la paire d'image 3 du chessboard n'est pas prise en compte 
    
"""
_________________________________________________________________________FINDING FUNDAMENTAL MATRIX_________________________________________________________________________
"""

"""
mtx est la matrice intrinsèque de la camera -> Matrice 3x3 -> matrice de calibration de caméra "K", permet l'ajustement et scaling
rvecs : est un tuple d'array qui contient les composantes rotationnelles
tvecs : idem mais pour translation
"""

ret,cameraMatrix,dist,rvecs,tvecs = calibration()


# pour i = 0,2,4 on a la matrice de projection de la camera de gauche (suite à l'ordre dans lequel les photos sont parcourues dans la calibration)
# pour i = 1,3,5 on a la matrice de projection de la camera de droite

def get_projection_matrix(i):
    R = cv2.Rodrigues(rvecs[i])[0]
    t = tvecs[i]
    Rt = np.concatenate([R,t], axis=-1) # [R|t] -> matrice de rotation
    P = np.matmul(cameraMatrix,Rt) # A[R|t]
    return P

def get_camera_centre_matrix(i): # centre optique d'une camera
    R = cv2.Rodrigues(rvecs[i])[0]
    t = tvecs[i]
    C = -R@t # est une 3x1, on ajoute une quatrième composante qu'on fixe à 1 pour une caméra centrée dans le repère world
    C = np.vstack([C,np.array([1])])
    return C

#Create a dictionnary containning all interesting matrixes
   
matrixes = dict()
matrixes['K'] = cameraMatrix
matrixes['rvecs'] = rvecs
matrixes['tvecs'] = tvecs

matrixes['P'],matrixes['C']= get_projection_matrix(0),get_camera_centre_matrix(1)
matrixes["P'"],matrixes["C'"]= get_projection_matrix(1),get_camera_centre_matrix(0)
matrixes["PseudoP"] = np.linalg.pinv(matrixes['P'])

# here we are only considerating the first set of 2 pictures
# print(get_projection_matrix(0))
# print()
# print(get_projection_matrix(1))
# print()
# print(get_projection_matrix(2))
# print()
# print(get_projection_matrix(3))

# # print(get_camera_centre_matrix(1))
# # print(get_camera_centre_matrix(0))
def crossMat(vec):# doit retourner
    x,y,z = vec
    M = np.array([[0,-z[0],y[0]],
                  [z[0],0,-x[0]],
                  [-y[0],x[0],0]])
    return M


#Compute first factor for Fundamental matrix

First = matrixes["P'"] @ matrixes['C']
First = crossMat(First)

#Compute second factor for Fundamental matrix

Second = matrixes["P'"]@ matrixes['PseudoP']

#Matrice Fondamentale

print("Fundamental")
F = First @ Second
print(F)


# # F = [P'C]x  P'  P+
# #     3x4 4x1 3x4 4x3
 
# #on a que l = F . x (l = ligne épipolaire, et x coordonnée du point image issue d'une image Left ou Right)


    




"""
_________________________________________________________________________EPILINES_________________________________________________________________________
"""


"""
APPLYING F on x coordinates to determine the epilines
"""



def display_lines(img,lines):
    """
    just used to check if scan_lines() found the wright points from the readlines in the scanLeft and scanRight images
    """
    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0))
        break


def get_redline_points(img_path):
    """
    get the points of the vertical redlines which are image points needed to calucalute epilines
    """
    img = cv2.imread(img_path)
    # imgR = cv2.imread('./scanRight/scan0003.png')

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set the lower and upper bounds for the red values
    lower_red = (0, 50, 50)
    upper_red = (10, 255, 255)

    # Create the mask using the cv22.inRange() function
    gray = cv2.inRange(hsv, lower_red, upper_red)

    lines = cv2.HoughLinesP(gray,1,np.pi/180,50,minLineLength=80,maxLineGap=300)
    # print(lines.shape) (6,1,4) on a dectecté 6 lignes , le 4 correspond aux quatres coordonnées x1,y1,x2,y2 
    lines = np.squeeze(lines)# pour retirer la dimension inutile
    return img,lines



def compute_epiline(x_coords): # x_coords : coordonnée du point dans le plan image
    x,y = x_coords
    a,b,c = F @ (np.array([[x,y,1]]).transpose()) # 0 ou 1 ?
    return a[0],b[0],c[0]


img,lines = get_redline_points(('./scanLeft/0003.png'))
display_lines(img,lines)

# print(lines[0])
x1,y1,x2,y2 = lines[0] # on récupère le premier point de la premiere ligne rouge trouvée 
print("First point left image coords :",(x1,y1))

a,b,c = compute_epiline((x1,y1))
#y = mx + q 
m = -a/b
q = -c/b
print("line",(m,q))


imgD = cv2.imread('./scanRight/scan0003.png')
cv2.line(imgD, (0, int(q)), (int(q/-m), 0), (0, 255, 0), 2)
# cv2.imshow('Droite', img)
cv2.imshow('Droite', imgD)
cv2.waitKey(0)
cv2.destroyAllWindows()





# for y in range(y1,y2+1):
#     a,b,c = F @ (np.array([[x1,y,0]]).transpose()) # 0 ou 1 ?
#     print(a,b,c)  
#     print((x1,y))    
#     y_min = 0
#     y_max = imgR.shape[0]
#     x_min = int((y_min * a + c) / -b)
#     x_max = int((y_max * a + c) / -b)
#     cv2.line(imgL, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     cv2.imshow('Droite', imgL)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break

# # cv2.imshow('Original', img)
# # cv2.imshow('Filtered', imgL)
# # cv2.imshow('Droite', imgR)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # LeftScans = glob.glob('./scanLeft/*.png')
# # RightScans =glob.glob('./scanRight/*.png')

# # lower_red = (0, 50, 50)
# # upper_red = (10, 255, 255)

# # for imgL,imgR in zip(LeftScans,RightScans):
# #     print(imgL,imgR)

# #     imgL = cv2.imread(imgL)
# #     imgR = cv2.imread(imgR)

# #     hsvL = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
# #     hsvR = cv2.cvtColor(imgR, cv2.COLOR_BGR2HSV)

# #     grayL = cv2.inRange(hsvL, lower_red, upper_red)
# #     grayR = cv2.inRange(hsvR, lower_red, upper_red)

# #     linesL = cv2.HoughLinesP(grayL,1,np.pi/180,50,minLineLength=8,maxLineGap=300)
# #     linesR = cv2.HoughLinesP(grayR,1,np.pi/180,50,minLineLength=8,maxLineGap=300)


# #     print(linesL)
# #     print(linesR)
# #     linesL = np.squeeze(linesL)
# #     linesR = np.squeeze(linesR)
# #     try:
# #         x1_L,y1_L,x2_L,y2_L = linesL[0]
# #         x1_R,y1_R,x2_R,y2_R = linesR[0]
# #         for y in range(y1_L,y2_L+1):
# #             #computing epiline
# #             a,b,c = F @ (np.array([[x1,y,1]]).transpose()) # 0 ou 1 ?
            
# #             # # Plot the epipolar line in the right image
# #             y_min = 0
# #             y_max = imgR.shape[0]
# #             x_min = int((y_min * a + c) / -b)
# #             x_max = int((y_max * a + c) / -b)
# #             # # cv2.line(right_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
# #             display_lines(imgR,(x_min, y_min, x_max, y_max))
# #             print(y)
# #             # print("epiline left image")
# #             #print((a,b,c))
# #             cv2.imshow('Original', imgL)
# #             cv2.imshow('Filtered', gray)
# #             cv2.waitKey(0)
# #             cv2.destroyAllWindows()
# #             break
# #         for y in range(y1_R,y2_R+1):
# #             # l = F @ (np.array([[x1,y,1]]).transpose()) # 0 ou 1 ?
# #             # Plot the epipolar line in the right image
# #             # y_min = 0
# #             # y_max = right_img.shape[0]
# #             # x_min = int((y_min * a + c) / -b)
# #             # x_max = int((y_max * a + c) / -b)
# #             # cv2.line(right_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
# #             # print("epiline right image")
# #             print(y)
# #             break
# #         break
# #     except Exception as e:
# #         print(e)



# # # Display the original and filtered images
# # # cv2.imshow('Original', img)
# # # cv22.imshow('Filtered', gray)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()