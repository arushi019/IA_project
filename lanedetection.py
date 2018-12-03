import os
import cv2
import numpy as np
# from conversion import *
from conversion2 import *
from imageai.Detection import ObjectDetection
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5")
detector.loadModel(detection_speed="flash")
def main():
    basedir = "2011_09_26/2011_09_26_drive_0015_sync/image_02/data"
    for root,dirs,files in os.walk(basedir):
        files.sort()
        for f in files:
            frame = cv2.imread(os.path.join(root,f))
            detections = detector.detectObjectsFromImage(input_image=os.path.join(root,f), output_image_path="object_detections/imagenew"+str(f)+".png")
            # convert(frame)
            total = findLanes(frame)

            total  = cv2.addWeighted(total, .5, cv2.imread("object_detections/imagenew"+str(f)+".png"),0.5, 0)
            cv2.imshow("xx",total)
            cv2.imwrite("detections/imagenew"+str(f)+".png",total)
            if( cv2.waitKey(1000) & 0xFF == ord('q')):
                break
if __name__ == '__main__':
    main()
