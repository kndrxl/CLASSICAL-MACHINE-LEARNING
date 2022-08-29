import numpy as np
import cv2

def generate_face_dataset(face_class, length, name):
    haar_data = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    capture = cv2.VideoCapture(0)
    data = []
    while True:
        flag, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if flag:
            faces = haar_data.detectMultiScale(gray)
            for x,y,w,h in faces:
                cv2.rectangle(gray,(x,y), (x+w, y+h), (255, 0, 255), 4)
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (50,50)) 
                print(len(data))
                if len(data) < length:
                    data.append(face)
            if cv2.waitKey(2) == 27 or len(data) == length: # 27 is ASCII number for 'Escape key'
                break
            
    capture.release()            
    cv2.destroyAllWindows()
    filename = f"dataset_generator/{face_class}/{name}.npy"
    np.save(filename, data)

    return {"Message": f"Data Saved at this path: {filename}"}

if __name__ == "__main__":
    print("\n**************** FIRST FACE ****************")
    length = int(input(f"\nEnter Number of face images to Capture: "))
    name = str(input(f"\nEnter the name of the 1st subject: "))
    face_class = "face_1"
    print("\n")
    print(generate_face_dataset(face_class, length, name))
    print("\n**************** SECOND FACE ****************")
    length = int(input(f"\nEnter Number of face images to Capture: "))
    name = str(input(f"\nEnter the name of the 2nd subject: "))
    face_class = "face_2"
    print("\n")
    print(generate_face_dataset(face_class, length, name))