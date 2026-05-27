import cv2 as cv
import multiprocessing as mp
import queue
from ultralytics import YOLO

CONF_THRESHOLD = 0.5
MODEL_PATH = "runs/detect/train2/weights/best.pt"

def predict(img_queue: mp.Queue, results_queue: mp.Queue):
    model = YOLO(MODEL_PATH)

    try:
        while True:
            img = img_queue.get()
            results_queue.put(list(model.predict(img, stream=True, conf=CONF_THRESHOLD)))
    except KeyboardInterrupt:
        pass

def main():
    cap = cv.VideoCapture("http://galaxy-s23:8080/video")

    img_queue = mp.Queue(1)
    results_queue = mp.Queue(1)

    process = mp.Process(target=predict, args=(img_queue, results_queue))
    processing = False
    results = None

    process.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if not processing:
                img_queue.put_nowait(frame)
                processing = True
            else:
                try:
                    results = results_queue.get_nowait()
                    processing = False
                except queue.Empty:
                    pass
            
            if results is not None:
                for result in results:
                    for cls, conf, xywh in zip(result.boxes.cls, result.boxes.conf, result.boxes.xywh):
                        cx, cy, bw, bh = xywh.tolist()
                        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                        label = f"{result.names[int(cls)]} {conf:.2f}"
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, label, (x1, y1 - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv.imshow("prediction", frame)

            key = cv.waitKey(1)
            if (key & 0xFF) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
        process.kill()
        process.join()
        img_queue.close()
        results_queue.close()

if __name__ == '__main__':
    main()
