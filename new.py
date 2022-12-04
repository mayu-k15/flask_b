import cv2
import matplotlib.pyplot as plt
import numpy as np


def predict_pothole():
    area=0
    yolo = cv2.dnn.readNet(
        "models/yolov4_tiny_pothole.cfg",
        "models/yolov4_tiny_pothole_last.weights")
    classes = []
    with open("pothole.names", 'r') as f:
        classes = f.read().splitlines()
    # print(len(classes))
    # img = cv2.imread("images/img_1.png")
    img = cv2.imread("img1.jpg")
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    # print(blob.shape)/
    i = blob[0].reshape(320, 320, 3)
    yolo.setInput(blob)
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layer_name)
    boxs = []
    confidences = []
    class_ids = []
    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxs.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # print("box",len(boxs))
    if len(boxs)!=0:
        indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxs), 3))
        for i in indexes.flatten():
            x, y, w, h = boxs[i]
            print("x "+str(x)+" y " + str(y)+" w " + str(w)+" h " + str(h)+"area " + str(w*h))

            area = (w*0.0264583333) * (h*0.0264583333)
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label + " " , (x, y + 20), font, 2, (255, 255, 255), 1)
        plt.imshow(img)
        plt.show()
        return area
    else:
        plt.imshow(img)
        plt.show()
        print("No pothole")
        return 0;


area=predict_pothole()
# print("area",area)


# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def predict_pothole(img):
#     yolo = cv2.dnn.readNet(
#         "D:\proje-tryout\Real-time-Pothole-Detector-and-Area-Calculator-main\models\yolov4_tiny_pothole.cfg",
#         "D:\proje-tryout\Real-time-Pothole-Detector-and-Area-Calculator-main\models\yolov4_tiny_pothole_last.weights")
#     classes = []
#     with open("pothole.names", 'r') as f:
#         classes = f.read().splitlines()
#     print(len(classes))
#     img = cv2.imread("images/images.jpg")
#     dimensions = img.shape
#     height = img.shape[0]
#     width = img.shape[1]
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     print(blob.shape)
#     i = blob[0].reshape(320, 320, 3)
#     plt.imshow(i)
#     yolo.setInput(blob)
#     output_layer_name = yolo.getUnconnectedOutLayersNames()
#     layeroutput = yolo.forward(output_layer_name)
#     boxs = []
#     confidences = []
#     class_ids = []
#     for output in layeroutput:
#         for detection in output:
#             score = detection[5:]
#             class_id = np.argmax(score)
#             confidence = score[class_id]
#             if confidence > 0.7:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxs.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#     print(len(boxs))
#     indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
#     font = cv2.FONT_HERSHEY_PLAIN
#     colors = np.random.uniform(0, 255, size=(len(boxs), 3))
#     for i in indexes.flatten():
#         x, y, w, h = boxs[i]
#         label = str(classes[class_ids[i]])
#         confi = str(round(confidences[i], 2))
#         color = colors[i]
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
#         cv2.putText(img, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 1)
#     # plt.imshow(img)
#     return
#
