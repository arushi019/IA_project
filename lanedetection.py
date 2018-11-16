import os
import cv2
import numpy as np
from conversion import *
from conversion2 import *
def main():
    basedir = "2011_09_26/2011_09_26_drive_0015_sync/image_02/data"
    for root,dirs,files in os.walk(basedir):
        for f in files:
            frame = cv2.imread(os.path.join(root,f))
            # convert(frame)
            findLanes(frame)
            # cv2.imshow("xx",frame)
            if( cv2.waitKey(1000) & 0xFF == ord('q')):
                break
if __name__ == '__main__':
    main()
