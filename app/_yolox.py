"""YOLOX ONNX inference via OpenCV's DNN module (no PyTorch required).

Vendored from OpenCV Zoo (https://github.com/opencv/opencv_zoo, Apache-2.0),
lightly adapted: the input blob is cast to float32 (required by OpenCV 5's
inference engine). This runs object detection with cv2.dnn only, so it works on
Python versions where PyTorch/ultralytics have no wheels yet.
"""
import cv2
import numpy as np


class YoloX:
    def __init__(self, modelPath, confThreshold=0.35, nmsThreshold=0.5, objThreshold=0.5,
                 backendId=0, targetId=0):
        self.num_classes = 80
        self.net = cv2.dnn.readNet(modelPath)
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)
        self.generateAnchors()

    def preprocess(self, img):
        blob = np.transpose(img, (2, 0, 1))
        return blob[np.newaxis, :, :, :].astype(np.float32)  # OpenCV 5 needs float32

    def infer(self, srcimg):
        input_blob = self.preprocess(srcimg)
        self.net.setInput(input_blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.postprocess(outs[0])

    def postprocess(self, outputs):
        dets = outputs[0]
        dets[:, :2] = (dets[:, :2] + self.grids) * self.expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.expanded_strides

        boxes = dets[:, :4]
        boxes_xywh = np.ones_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xywh[:, 2] = boxes[:, 2]
        boxes_xywh[:, 3] = boxes[:, 3]

        scores = dets[:, 4:5] * dets[:, 5:]
        max_scores = np.amax(scores, axis=1)
        max_scores_idx = np.argmax(scores, axis=1)

        keep = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), max_scores.tolist(), max_scores_idx.tolist(),
            self.confThreshold, self.nmsThreshold,
        )
        candidates = np.concatenate(
            [boxes_xywh, max_scores[:, None], max_scores_idx[:, None]], axis=1
        )
        if len(keep) == 0:
            return np.array([])
        return candidates[keep]

    def generateAnchors(self):
        self.grids = []
        self.expanded_strides = []
        hsizes = [self.input_size[0] // s for s in self.strides]
        wsizes = [self.input_size[1] // s for s in self.strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            self.grids.append(grid)
            shape = grid.shape[:2]
            self.expanded_strides.append(np.full((*shape, 1), stride))
        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)
