
import flask
import werkzeug
import base64
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

app = flask.Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    area = 0
    yolo = cv2.dnn.readNet(
        "models/yolov4_tiny_pothole.cfg",
        "models/yolov4_tiny_pothole_last.weights")
    classes = []
    with open("pothole.names", 'r') as f:
        classes = f.read().splitlines()
    # print(len(classes))
    img = cv2.imread(imagefile.filename)
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
    if len(boxs) != 0:
        indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxs), 3))
        for i in indexes.flatten():
            x, y, w, h = boxs[i]
            print("x " + str(x) + " y " + str(y) + " w " + str(w) + " h " + str(h) + "area " + str(w * h))
            area = w * h
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label + " ", (x, y + 20), font, 2, (255, 255, 255), 1)
        print(area)
        return str(area)
    else:
        return "1"

app.run(host="0.0.0.0", port=5000, debug=True)


# import base64
# import io
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import werkzeug
# from PIL import Image
# from flask import Flask, request
#
# app = Flask(__name__)
#
#
# # @app.route('/')
# # def hello_world():
# #     print(__name__)
# #     return 'Hello Everyone'
#
#
# @app.route('/one')
# def hello_one():
#     return 'Hello one'
#
#
# @app.route('/two')
# def hello_two():
#     return 'Hello two'
#
#
# @app.route('/',methods = ['GET', 'POST'])
# def main_interface():
#     imagefile = Flask.request.files['image']
#     filename = werkzeug.utils.secure_filename(imagefile.filename)
#     print("\nReceived image File name : " + imagefile.filename)
#     imagefile.save(filename)
#     return "Image Uploaded Successfully"
#
#     # imagefile = Flask.request.files['image0']
#     # filename = werkzeug.utils.secure_filename(imagefile.filename)
#     # print("\nReceived image File name : " + imagefile.filename)
#     # imagefile.save(filename)
#     #
#     # img = scipy.misc.imread(filename, mode="L")
#     # img = img.reshape(784)
#     # loaded_model = keras.models.load_model('model.h5')
#     # predicted_label = loaded_model.predict_classes(numpy.array([img]))[0]
#     # print(predicted_label)
#     #
#     # return str(predicted_label)
#
#
#     # area = 0
#     # yolo = cv2.dnn.readNet(
#     #     "models/yolov4_tiny_pothole.cfg",
#     #     "models/yolov4_tiny_pothole_last.weights")
#     # classes = []
#     # with open("pothole.names", 'r') as f:
#     #     classes = f.read().splitlines()
#     # # print(len(classes))
#     # img = cv2.imread("image/img2.jpg")
#     # dimensions = img.shape
#     # height = img.shape[0]
#     # width = img.shape[1]
#     # blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
#     # # print(blob.shape)/
#     # i = blob[0].reshape(320, 320, 3)
#     # yolo.setInput(blob)
#     # output_layer_name = yolo.getUnconnectedOutLayersNames()
#     # layeroutput = yolo.forward(output_layer_name)
#     # boxs = []
#     # confidences = []
#     # class_ids = []
#     # for output in layeroutput:
#     #     for detection in output:
#     #         score = detection[5:]
#     #         class_id = np.argmax(score)
#     #         confidence = score[class_id]
#     #         if confidence > 0.7:
#     #             center_x = int(detection[0] * width)
#     #             center_y = int(detection[1] * height)
#     #             w = int(detection[2] * width)
#     #             h = int(detection[3] * height)
#     #
#     #             x = int(center_x - w / 2)
#     #             y = int(center_y - h / 2)
#     #
#     #             boxs.append([x, y, w, h])
#     #             confidences.append(float(confidence))
#     #             class_ids.append(class_id)
#     # # print("box",len(boxs))
#     # if len(boxs) != 0:
#     #     indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
#     #     font = cv2.FONT_HERSHEY_PLAIN
#     #     colors = np.random.uniform(0, 255, size=(len(boxs), 3))
#     #     for i in indexes.flatten():
#     #         x, y, w, h = boxs[i]
#     #         print("x " + str(x) + " y " + str(y) + " w " + str(w) + " h " + str(h) + "area " + str(w * h))
#     #         area = w * h
#     #         label = str(classes[class_ids[i]])
#     #         confi = str(round(confidences[i], 2))
#     #         color = colors[i]
#     #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
#     #         cv2.putText(img, label + " ", (x, y + 20), font, 2, (255, 255, 255), 1)
#     #     print(area)
#     #     return str(area)
#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')
#
#
#     # else:
#     #     plt.imshow(img)
#     #     plt.show()
#     #     print("No pothole")
#     #     return '0';
#
# # import base64
# # import io
# # import cv2
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from PIL import Image
# # from flask import Flask, request
# #
# # app = Flask(__name__)
# #
# #
# # # @app.route('/')
# # # def hello_world():
# # #     print(__name__)
# # #     return 'Hello Everyone'
# #
# #
# # @app.route('/one')
# # def hello_one():
# #     return 'Hello one'
# #
# #
# # @app.route('/two')
# # def hello_two():
# #     return 'Hello two'
# #
# #
# # @app.route('/')
# # def main_interface():
# #     area = 0
# #     yolo = cv2.dnn.readNet(
# #         "models/yolov4_tiny_pothole.cfg",
# #         "models/yolov4_tiny_pothole_last.weights")
# #     classes = []
# #     with open("pothole.names", 'r') as f:
# #         classes = f.read().splitlines()
# #     # print(len(classes))
# #     img = cv2.imread("image/img2.jpg")
# #     dimensions = img.shape
# #     height = img.shape[0]
# #     width = img.shape[1]
# #     blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
# #     # print(blob.shape)/
# #     i = blob[0].reshape(320, 320, 3)
# #     yolo.setInput(blob)
# #     output_layer_name = yolo.getUnconnectedOutLayersNames()
# #     layeroutput = yolo.forward(output_layer_name)
# #     boxs = []
# #     confidences = []
# #     class_ids = []
# #     for output in layeroutput:
# #         for detection in output:
# #             score = detection[5:]
# #             class_id = np.argmax(score)
# #             confidence = score[class_id]
# #             if confidence > 0.7:
# #                 center_x = int(detection[0] * width)
# #                 center_y = int(detection[1] * height)
# #                 w = int(detection[2] * width)
# #                 h = int(detection[3] * height)
# #
# #                 x = int(center_x - w / 2)
# #                 y = int(center_y - h / 2)
# #
# #                 boxs.append([x, y, w, h])
# #                 confidences.append(float(confidence))
# #                 class_ids.append(class_id)
# #     # print("box",len(boxs))
# #     if len(boxs) != 0:
# #         indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
# #         font = cv2.FONT_HERSHEY_PLAIN
# #         colors = np.random.uniform(0, 255, size=(len(boxs), 3))
# #         for i in indexes.flatten():
# #             x, y, w, h = boxs[i]
# #             print("x " + str(x) + " y " + str(y) + " w " + str(w) + " h " + str(h) + "area " + str(w * h))
# #             area = w * h
# #             label = str(classes[class_ids[i]])
# #             confi = str(round(confidences[i], 2))
# #             color = colors[i]
# #             cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
# #             cv2.putText(img, label + " ", (x, y + 20), font, 2, (255, 255, 255), 1)
# #         print(area)
# #         return str(area)
# #
# # if __name__ == "__main__":
# #     app.run(host='0.0.0.0')
# #
# #
# #     # else:
# #     #     plt.imshow(img)
# #     #     plt.show()
# #     #     print("No pothole")
# #     #     return '0';