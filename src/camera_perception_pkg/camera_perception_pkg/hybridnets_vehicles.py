"""Vehicle boxes from HybridNets (detection head only; its lane/road outputs unused).

The model file is either a TorchScript export of the official weights (hybridnets.pth ->
*.torchscript.pt, runs on the GPU when CUDA is available, else CPU) or the ONNX model (CPU).

Pre/post-processing follows ONNX-HybridNets-Multitask-Road-Detection by Ibai Gorordo (MIT License,
Copyright (c) 2022 Ibai), as used in YuriPark0123/skku_autonomous_driving where this model detected
the track's cars well on CPU.
"""

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def transform_boxes(boxes, anchors):
    """Regression output (dy, dx, dh, dw per anchor) -> (xmin, ymin, xmax, ymax) in input pixels."""
    y_centers_a = (anchors[:, 0] + anchors[:, 2]) / 2
    x_centers_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]
    w = np.exp(boxes[:, 3]) * wa
    h = np.exp(boxes[:, 2]) * ha
    y_centers = boxes[:, 0] * ha + y_centers_a
    x_centers = boxes[:, 1] * wa + x_centers_a
    return np.vstack((x_centers - w / 2, y_centers - h / 2, x_centers + w / 2, y_centers + h / 2)).T


def nms(boxes, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        x0 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.maximum(0, x1 - x0 + 1) * np.maximum(0, y1 - y0 + 1)
        iou = inter / (areas[best] + areas[rest] - inter)
        order = rest[iou < iou_threshold]
    return boxes[keep], scores[keep]


class HybridNetsVehicles:
    def __init__(self, model_path, anchor_path, conf=0.5, iou=0.5, threads=2, device='auto',
                 input_size=(384, 512)):
        self.anchors = np.squeeze(np.load(anchor_path))
        self.conf, self.iou = conf, iou
        if str(model_path).endswith('.onnx'):
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = threads  # leave CPU for the rest of the pipeline
            self.session = onnxruntime.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            self.input_height, self.input_width = model_input.shape[2], model_input.shape[3]
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.provider = 'cpu (onnxruntime)'
            self.run = self.run_onnx
        else:
            import torch
            self.torch = torch
            if device == 'auto':
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if device == 'cpu':
                torch.set_num_threads(threads)
            self.device = device
            self.model = torch.jit.load(model_path, map_location=device).eval()
            self.input_height, self.input_width = input_size
            self.provider = f'{device} (torch)'
            self.run = self.run_torch

    def run_onnx(self, tensor):
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return (outputs[self.output_names.index('regression')],
                outputs[self.output_names.index('classification')])

    def run_torch(self, tensor):
        with self.torch.inference_mode():
            regression, classification = self.model(self.torch.from_numpy(tensor).to(self.device))
        return regression.float().cpu().numpy(), classification.float().cpu().numpy()

    def detect(self, bgr):
        """[(xmin, ymin, xmax, ymax), score] in image pixels."""
        height, width = bgr.shape[:2]
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        tensor = ((rgb / 255.0 - MEAN) / STD).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        regression, classification = self.run(tensor)
        scores = classification[0, :, 0]
        keep = scores > self.conf
        if not keep.any():
            return []
        boxes, scores = transform_boxes(regression[0][keep], self.anchors[keep]), scores[keep]
        boxes[:, [0, 2]] *= width / self.input_width
        boxes[:, [1, 3]] *= height / self.input_height
        boxes, scores = nms(boxes, scores, self.iou)
        return [(tuple(float(v) for v in b), float(s)) for b, s in zip(boxes, scores)]
