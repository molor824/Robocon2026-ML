import os

def latest_train_path():
    trains = os.listdir("runs/detect")
    return os.path.join("runs/detect", max(trains))

def load_best():
    return os.path.join(latest_train_path(), "weights/best.pt")
def load_last():
    return os.path.join(latest_train_path(), "weights/last.pt")
