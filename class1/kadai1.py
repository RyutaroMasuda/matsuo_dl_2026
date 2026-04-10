
from ultralytics import YOLO
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='YOLOのデモ')
parser.add_argument('--yolo_version', type=str, help='sかnのどちらかの数字を指定',default='s')

args = parser.parse_args()
if args.yolo_version == 's':
    model_path = 'yolov8s.pt'
elif args.yolo_version == 'n':
    model_path = 'yolov8n.pt'
else:
    raise ValueError("yolo_versionはsかnを指定のこと")

def load_yolo_model(model_path=model_path):
    model = YOLO(model_path)
    return model

def detect_objects_in_image(model, image_path):
    results = model(image_path)
    return results

def visualize_results(image_path, results):
    img = cv2.imread(image_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]          # クラス名を取得
            conf = float(box.conf[0])               # 信頼度を取得
            label = f"{cls_name} {conf:.2f}"        # 例: "person 0.87"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, cls_name, (int(x1), int(y1) - 20),   # クラス名：ボックス上
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (int(x1), int(y2) + 25),  # 信頼度：ボックス下
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Result', img)
    cv2.imwrite(f'output_yolov8{args.yolo_version}.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# modelのロード
model = load_yolo_model()
print(f"モデルがロードされました: {model}")

# オブジェクトの個数検出
image_path = './class1/statue_auditorium_eyecatch.jpg'
results = detect_objects_in_image(model, image_path)
print(f"検出された物体: {len(results[0].boxes)} 個")

# 可視化
visualize_results(image_path, results)