"""
YOLOv7 Classification — Placeholder

NOTE: YOLOv7 (WongKinYiu/yolov7) was designed for object detection only.
It does NOT have a built-in classification mode or pretrained -cls weights.

For the thesis comparison, classification is performed using:
  - YOLOv8-cls (Ultralytics, native classification)
  - YOLOv11-cls (Ultralytics, native classification)

If you want to add YOLOv7 to the comparison, possible approaches:
  1. Use the YOLOv7 backbone as a feature extractor + add a classification head
  2. Fine-tune the backbone with a standard PyTorch training loop
  3. Compare YOLOv7 detection performance on an annotated version of the dataset

This file is kept as a placeholder for future extension.
"""


def main():
    print("YOLOv7 does not support classification natively.")
    print("Use train_ultralytics.py for YOLOv8-cls and YOLOv11-cls models.")


if __name__ == "__main__":
    main()
