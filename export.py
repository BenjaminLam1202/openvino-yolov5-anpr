from ultralytics import YOLO

# Load a model
model = YOLO("../../license-plate-recognition/weights/plate_yolov8n.pt")  # load a pretrained model (recommended for training)

success = model.export(format="openvino")  # export the model to ONNX format