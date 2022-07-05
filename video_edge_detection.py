import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()
    cv.imshow('Camera',frame)


    #live video edge detectation using laplacian filter(usually very noisy image )
    laplacian = cv.Laplacian(frame,cv.CV_64F)
    laplacian = np.uint8(laplacian)
    cv.imshow('Laplacian',laplacian)
    
    #live video edge detectation using canny filter (usually less noisy image depends in the threshold)
    edges = cv.Canny(frame,100,100)
    cv.imshow('Canny', edges)


    if cv.waitKey(5) == ord("x"):
        break

camera.release()
cv.destroyAllWindows()