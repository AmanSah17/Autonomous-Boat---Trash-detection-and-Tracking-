# Autonomous-Boat---Trash-detection-and-Tracking-


This yolo__preds_angle.ipynb file calls for yolo_preds_2.py , file to generate bounding boxes and class probability scores.
We have used YOLO-V5 CNN architecture to train the model on more than 10K+ trash objects floating on water.
After training on Google Colab , the best.pt model is converted to .ONNX format for better compatibility with OpenCV.



yolo_preds_angle.ipynb file will generated the correponding ounding boxes, class probability score , centroid of the object, as well as the angle it's centroid making at any moment of time(Frame Wise) with respect to X-Y (2D-coordinate System). This helps the autonomous Vehicle to identify the accurate location of the object in the 2D Map.
