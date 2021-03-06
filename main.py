import cv2
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse

def detect(frame): 
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (4, 4), scale = 1.03)
    person = 1
    # x, y are the starting coordinates and w, h are width and height of box respectively
    bounding_box_cordinates = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bounding_box_cordinates])
    # applying the Non Maximum Suppression(NMS) algorithm
    pick = non_max_suppression(bounding_box_cordinates, probs=None, overlapThresh=0.37)
    # drawing the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        person += 1
    # adding text on the output screen
    cv2.putText(frame, 'Status : Detected', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2)
    cv2.putText(frame, f'Total persons: {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2)
    cv2.imshow('Output', frame)
    return frame

# Detecting persons in a video
def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()
        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1)
            # On pressing 'q' key the video will close  
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

# Detecting from a webcam
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()

# Detecting persons from an image
def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    image = imutils.resize(image, width = min(800, image.shape[1])) 
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# checks if input is an image/ video/ webcam
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False
    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
    if camera:
        print('Opening WebCam...')
        detectByCamera(writer)
    elif video_path is not None:
        print('Opening Video from path...')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('Opening Image from path...')
        detectByPathImage(image_path, args['output'])

def argsParser():
    # calling the command-line argument parsing module
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File")
    arg_parse.add_argument("-c", "--camera", default=False, help="true if cam is used")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())
    return args

if __name__ == "__main__":
    # initializing the HOG descriptor
    HOGCV = cv2.HOGDescriptor()
    # setting the support vector machine (SVM) to be pre-trained
    # calling the pre-trained model for human detection 
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # calling the argParser for passed the arguments in the terminal
    args = argsParser()
    humanDetector(args)
