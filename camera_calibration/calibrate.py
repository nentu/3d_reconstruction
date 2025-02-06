import cv2 as cv
from tqdm import tqdm
import numpy as np
import os

def get_calib_param(camera_name, cam_id):

    root = f'./data/{camera_name}/'


    if not os.path.exists(root+"res.npz"):

        cap = cv.VideoCapture(cam_id)



        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        size = (5, 6)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        while True:
            # Capture frame-by-frame
            ret, img = cap.read()
        
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, size, None)
        
            # If found, add object points, image points (after refining them)
            if ret == True:
                cv.imwrite(root + f"img_{len(os.listdir(root))}.jpg", img)
                cv.drawChessboardCorners(img, size, corners, ret)
                cv.imshow('frame', img)
                cv.waitKey(500)
            else:
                # Display the resulting frame
                cv.imshow('frame', img)

            if cv.waitKey(1) == ord('q'):
                break

        for fname in os.listdir(root):
            if '.img' not in fname:
                continue
            img = cv.imread(root + fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, size, None)

        
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
        
                corners2 = cv.cornerSubPix(gray,corners, (sum(size),sum(size)), (-1,-1), criteria)
                imgpoints.append(corners2)
        
        cv.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        np.savez(root+"res.npz",
            ret = ret,
            mtx = mtx,
            dist = dist,
            rvecs = rvecs,
            tvecs = tvecs,
        )

    return np.load(root+"res.npz")

class Calibrator:
    mtx: np.array
    dist: np.array
    newcameramtx: np.array
    roi: np.array
    name: str

    def __init__(self, name, camera_id, w, h):
        a = get_calib_param(name, camera_id)

        self.mtx = a['mtx']
        self.dist = a['dist']

        self.newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

    def calibrate(self, img):
        dst = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        
        # crop the image
        x, y, w, h = self.roi
        return dst[y:y+h, x:x+w]

if __name__ == '__main__':

    cap = cv.VideoCapture(1)
    # Capture frame-by-frame
    ret, img = cap.read()
    h,  w = img.shape[:2]

    calibrator = Calibrator("laptop", 1, w, h)

    while True:
        ret, img = cap.read()

        dst = calibrator.calibrate(img)

        cv.imshow('orig', img)
        cv.imshow('calibresult', dst)

        
        if cv.waitKey(1) == ord('q'):
            break
        

