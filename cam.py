import cv2
import torch
from ultralytics import YOLO

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')  #Replace 'mps' with your GPU value

    model = YOLO('/Users/advaitpatil/Documents/final pp/best.pt') #Replace the path to where your model path is in your computer or storage space
    model.to(device)

    model.predict(source=torch.zeros(1, 3, 640, 640).to(device), 
                  imgsz=640, verbose=False, half=True)

    cam = cv2.VideoCapture(0)    #The default value is 0, if you want to use another camera, replace the value of '0' with the camera value
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("Camera not found")
        return

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=640,      #Replace this with your image training size, my trained model size is 640
            conf=0.4,      #The 'conf' value shows the minimum confidence level the AI model should have to actually detect the fire as "fire" (this is in decimals e.g. 0.6 = 60% confidence level)
            half=True,
            verbose=False,      #Set to 'True' if you want logging
            device=device
        )

        cv2.imshow("Detection", results[0].plot())      #Remove this line out if you don't want to display; replace 'results[0].plot()' with your targeted capture or display

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()