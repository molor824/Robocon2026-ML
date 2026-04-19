from ultralytics import YOLO
from model_loader import load_last

def main():
    # try:
    #     path = load_last()
    #     print(f"Starting from last weight {path}")
    # except:
        # path = "yolo12n.pt"
        # print(f"Starting from new weight {path}")

    model = YOLO("yolo26n.pt")
    model.train(data="data.yaml", epochs=100, imgsz=640, save=True, batch=-1)

if __name__ == "__main__":
    main()