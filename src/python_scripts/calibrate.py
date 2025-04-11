import numpy as np
import cv2 as cv
import glob
import robotpy_apriltag
import wpimath.geometry

    
if __name__ == "__main__":
    
    # initialize the april tag
    tag = robotpy_apriltag.AprilTagDetector()
    tag.addFamily("tag36h11", 3)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((2*2,3), np.float32)
    objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
    print(objp)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        l: list[robotpy_apriltag.AprilTagDetection] = tag.detect(gray)
        if (len(l) > 0):
            print(l[0])
            estimator = robotpy_apriltag.AprilTagPoseEstimator()
            pose = estimator.estimate(l[0])
            print(pose)
            
        # Break the loop if 'q' key is pressed
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()