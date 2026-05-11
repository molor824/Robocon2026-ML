import cv2 as cv
import os

def listdir_full(path):
    return [os.path.join(path, p) for p in os.listdir(path)]

def main():
    IMG_PATH = listdir_full("../Robocon2026Simulation/.datasets/images/test")
    LABEL_PATH = [p.replace("images", "labels").replace(".png", ".txt") for p in IMG_PATH]

    try:
        for img_path, label_path in zip(IMG_PATH, LABEL_PATH):
            img = cv.imread(img_path)
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            h, w = img.shape[:2]
            
            for line in lines:
                cls, xc, yc, bw, bh = map(float, line.split())

                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)

                cls_id = int(cls)
                cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv.putText(img, str(cls_id), (xmin, ymin - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv.imshow("Dataset check", img)
            key = cv.waitKey(0)
            if key == 27:
                break
    finally:
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
