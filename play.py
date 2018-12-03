import cv2
import os
basedir = "detections/"
for root,dirs,files in os.walk(basedir):
    files.sort()
    for f in files:
        frame = cv2.imread(os.path.join(root,f))
        cv2.imshow("xx",frame)
        if( cv2.waitKey(110) & 0xFF == ord('q')):
            break
