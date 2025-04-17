import cv2
import pytesseract
from pytesseract import image_to_string
import numpy as np
import re
from collections import Counter

from crop_tool import crop_image
from crop_tool import find_bound_box
from crop_tool import rotate_image

'''
OTSU vs GLOBAL

- Global handles the margins of the dial better, leading to less of the number cutoff. But 

- However glare noise is markedly reduced in Otsu 

- There are some strange cases where '5' is confused for 0
'''


def blur(img):
    median = cv2.medianBlur(img,9)
    #norm = cv2.blur(img,(9,9))

    # cv2.imshow("norm",norm)

    return median


def preprocessing(img,debug=False,still=False):
    #grayscale
    if img is None or img.size == 0:
         return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = blur(img)

    _, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if debug:
        cv2.imshow("preprocssed",otsu)
    if still:
        cv2.waitKey(0)
    
    # NOTE: comment these two lines out in order to display processed results

    return otsu

def read_img(img):

    input = preprocessing(img)

    if img is None:
         return ""
    data = pytesseract.image_to_string(input,config='--psm 8 --oem 3 -l digits -c tessedit_char_whitelist=.0123456789 -c load_system_dawg=0 -c load_freq_dawg=0')
    
    filtered_data = re.sub(r'[^0-9.]', '', data)

    # heuristic corrections

    # if size of just 3, add a . , if size of just 4, replace second to last with .
    # print(filtered_data)
    if (len(filtered_data) == 3):
            filtered_data = filtered_data[:2] + '.' + filtered_data[2:]
            
    elif (len(filtered_data) == 4):
                filtered_data = f"{filtered_data[:2]}.{filtered_data[3:]}"
    
    # NOTE: to see char results from OCR uncomment HERE
    # print(filtered_data)

    return filtered_data

def perform_detection(cap : cv2.VideoCapture):
    if not cap.isOpened():
        print("Camera not found or not accessible")
        exit(1)

    for i in range(0,2):
        # NOTE: rn this doesn't do any error handling
        # capture 10 frames and store into a list 
        frames = [cap.read()[1] for i in range(0,10)]

        # ASSUME: frames are captured within same time frame, thus same position

        # take one frame and get bounding box, store as variable use later
        bbox, skew = find_bound_box(frames[0])

        if skew != 0.0:
            frames = list(map(lambda frame: rotate_image(frame,skew), frames))

        # crop 10 frames 
        cropped_frames = map(lambda frame: crop_image(frame,bbox), frames)

        # run read_img() and store return results into a list
        res = map(read_img,  cropped_frames)

        # check for consensus by first creating a Counter 
        result_count = Counter(res)
        
        # then call max() of that counter object 
        consensus = max(result_count, key=result_count.get)

        # finally look at the frequency and if it passes a threshold then return
        if (result_count.get(consensus) >= 8):
            return consensus


    #return empty
    return ""

# run video feed and 

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found or not accessible")
        exit(1)
    
    # calibrate camera

    #config 
    try:
        i = 0
        while True:
                ret, frame = cap.read()

                #for now frame is our image
                

                crop_frame, frame = crop_image(frame)

                if not ret: 
                    print("Failed to grab frame")
                    break
                
                processed = preprocessing(frame)
                filtered_data = read_img(crop_frame)
                print(f'\rFiltered data: {filtered_data}', end='', flush=True)

                cv2.imshow('Video feed',frame)
                cv2.imshow('processed',processed)
            # cv2.imshow('crop feed', crop_frame)

            # exit check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()