from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel(detection_speed="flash")

detections = detector.detectObjectsFromImage(input_image="image2.png", output_image_path="imagenew.jpeg")

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
