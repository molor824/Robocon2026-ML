from ultralytics import YOLO
import os

RUNS_DIR = "runs/detect"
LAST_PATH= "weights/last.pt"

run_dirs = os.listdir(RUNS_DIR)
run_dirs.sort()
run_dirs = [os.path.join(RUNS_DIR, d, LAST_PATH) for d in run_dirs]

path = "yolo12n.pt"

if len(run_dirs) != 0:
    path = run_dirs[-1]
    print(f"Starting from last weight {path}")

model = YOLO(path)
model.train(data="data.yaml", epochs=100, imgsz=640, save=True, batch=-1)
