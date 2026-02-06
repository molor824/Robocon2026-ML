import cv2 as cv
from ultralytics import YOLO
import os
import random

if __name__ == '__main__':
    # read random image
    PATH = "../Robocon2026Simulation/.datasets/images/val"

    images = os.listdir(PATH)

    while True:
        image_index = random.randint(0, len(images) - 1)

        img = cv.imread(f"{PATH}/{images[image_index]}")

        model = YOLO("runs/detect/train/weights/best.pt")

        result = model(img)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            conf = float(box.conf[0])

            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"

            cv.putText(img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv.imshow("Result", img)

        if cv.waitKey(0) & 0xFF == ord('q'):
            break
