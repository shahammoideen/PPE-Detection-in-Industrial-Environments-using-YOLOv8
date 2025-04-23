# PPE-Detection-in-Industrial-Environments-using-YOLOv8
This project implements a two-stage object detection pipeline for ensuring safety compliance in industrial settings. Using YOLOv8, the system first detects persons in an image and then performs a secondary detection for Personal Protective Equipment (PPE) — such as helmets, vests, and gloves — within those regions. The solution simulates a real-world AI safety inspection system to enhance automation in hazardous environments.

Core Features:

  Two-Stage Object Detection: Person detection followed by PPE detection on cropped ROIs.

  Annotation Conversion: Converts PascalVOC to YOLO format.

  Custom Dataset Handling: Efficient folder structure for training YOLOv8.

  YOLOv8 Training: Separate models trained for persons and PPE.

  ONNX Export: Lightweight inference-ready model format for deployment.

  End-to-End Inference Pipeline: Full implementation for real-time predictions.
