import os
import cv2
import numpy as np
import collections as colls


class YoloCocoNet:
    def __init__(self, path_to_directory):
        """
        :param path_to_directory: directory where 'yolov3.weights', 'yolov3.cfg' and 'coco.names' are
        """

        def does_file_exists(path):
            if not os.path.isfile(path):
                raise FileExistsError(path)

        path_to_weights = os.path.sep.join(
            [path_to_directory, "yolov3.weights"])
        does_file_exists(path_to_weights)
        path_to_config = os.path.sep.join([path_to_directory, "yolov3.cfg"])
        does_file_exists(path_to_config)
        path_to_labels = os.path.sep.join([path_to_directory, "coco.names"])
        does_file_exists(path_to_labels)
        self.labels = open(path_to_labels).read().strip().split("\n")

        np.random.seed(42)
        self.label_colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8")

        self.net = cv2.dnn.readNetFromDarknet(path_to_config, path_to_weights)

        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i[0] - 1]
                            for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame, min_confidence, min_nms_threshold):
        """
        Takes an BGR (!!!) image as an input and returns list of namedtuples with such fields:
        - box (of detected object)
        - confidence (0..1: how probable in dnn's points of view that in that box is the objects dnn 'says')
        - class_id (there are lots of classes the net can detect; see coco.names)

        :param frame: BGR frame
        :param min_confidence:
        :param min_nms_threshold: non maxima suppression threshold
        :return: list of Detected_obj_inf (see definition below)
        """

        if len(frame.shape) != 3:
            raise Exception('Frame HAS TO be BGR')

        # the model was trained to detect objects on image with this size
        destination_image_shape = (416, 416)
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, destination_image_shape, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > min_confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    frame_height, frame_width = frame.shape[:2]
                    box = detection[0:4] * np.array(
                        [frame_width, frame_height, frame_width, frame_height])
                    center_x, center_y, width, height = box.astype("int")

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes_of_sure_objects = cv2.dnn.NMSBoxes(
            boxes, confidences, min_confidence, min_nms_threshold)
        if len(indexes_of_sure_objects):
            boxes = [boxes[i] for i in indexes_of_sure_objects.flatten()]
            confidences = [confidences[i]
                           for i in indexes_of_sure_objects.flatten()]
            class_ids = [class_ids[i]
                         for i in indexes_of_sure_objects.flatten()]
        else:
            boxes.clear()
            confidences.clear()
            class_ids.clear()

        Detected_obj_inf = colls.namedtuple(
            'Detected_obj_inf', ['box', 'confidence', 'class_id'])
        detected_objects = [Detected_obj_inf(
            boxes[i], confidences[i], class_ids[i]) for i in range(len(boxes))]
        return detected_objects

    def print_resulting_boxes(self, frame, detected_objects):
        for detected_object in detected_objects:
            # x, y -- top left corner

            x, y = detected_object.box[0], detected_object.box[1]
            w, h = detected_object.box[2], detected_object.box[3]

            color = [int(c)
                     for c in self.label_colors[detected_object.class_id]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(
                self.labels[detected_object.class_id], detected_object.confidence)
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
