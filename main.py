import cv2 as cv
from tkinter import *

def age_gender():
    def faceBox(faceNet,frame):

        frameWidth=frame.shape[1]
        frameHeight=frame.shape[0]
        blob = cv.dnn.blobFromImage(frame, 1.0,(227,227), [104,117,123], swapRB = False)
        faceNet.setInput(blob)
        detection = faceNet.forward()
        bboxs=[]
        for i in range (detection.shape[2]):
            confidence = detection[0,0,i,2]
            if confidence>0.7:
                x1=int(detection[0,0,i,3]*frameWidth)
                y1=int(detection[0,0,i,4]*frameHeight)
                x2=int(detection[0,0,i,5]*frameWidth)
                y2=int(detection[0,0,i,6]*frameHeight)
                bboxs.append([x1,y1,x2,y2])
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        return frame,bboxs

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"


    faceNet = cv.dnn.readNet(faceModel, faceProto)
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel,genderProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']


    video = cv.VideoCapture(0)

    while True:
        ret,frame = video.read()
        frame, bboxs = faceBox(faceNet, frame)
        for bbox in bboxs:
            face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            blob = cv.dnn.blobFromImage(face, 1.0,(227,227), MODEL_MEAN_VALUES, swapRB = False)
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]


            label="{},{}".format(gender, age)
            cv.putText(frame, label, (bbox[0],bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv.LINE_AA)


        cv.imshow("Age_Gender",frame)
        k=cv.waitKey(1)

        
        if k == ord('x'):
            break
    video.release()
    cv.destroyAllWindows()


def motion():
    camera = cv.VideoCapture(0)

    _, frame1 = camera.read()
    _, frame2 = camera.read()

    while True:

        #live video motion detection using contours
        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)
        _, thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
        dilated  = cv.dilate (thresh, None, iterations = 3)
        contours, _ = cv.findContours(dilated,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour)>200:
                cv.drawContours(frame1, contours, -1, (0,0,255),2)
                cv.putText(frame1,'Status: {}'.format('Movement'),(10,40),cv.FONT_ITALIC,1,(0,0,255), 3)
            cv.imshow('Camera',frame1)
        frame1 = frame2
        _, frame2 = camera.read()

        if cv.waitKey(5) == ord("x"):
            break

    camera.release()
    cv.destroyAllWindows()


window = Tk()
window.title('MOTION, AGE & GENDER DETECTOR')
label = Label(window, width = 50, height = 10)
label.pack(side = TOP)

lframe = Frame(window)
lframe.pack()
label1 = Label(lframe,text="MOTION, AGE & GENDER",font="times 30",pady=10)
label1.pack(side=TOP)
label2 = Label(lframe,text="DETECTOR",font="times 30",pady=10)
label2.pack(side = BOTTOM)


bframe = Frame(window)
bframe.pack()
b1 = Button(bframe,text = 'MOTION DETECTION', command = motion,pady=5)
b1.config(font=( 12))
b1.pack()
b2 = Button(bframe,text = 'AGE & GENDER DETECTION', command = age_gender,pady=5)
b2.config(font=(12))
b2.pack()
b3 = Button(bframe,text = 'EXIT',width = 15, command = window.destroy,pady=5)
b3.config(font=(12))
b3.pack()

frame = Frame(window)
frame.pack()
label3 = Label(frame,text="!!!!To close the detectors press 'x' on the keyboard !!!!",font="times 20",pady=10,padx= 5)
label3.pack(side=TOP)



window.mainloop()