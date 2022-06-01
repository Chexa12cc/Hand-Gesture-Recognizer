# HAND GESTURE PROJECT
# IT CAN BE TRAINED FOR ANY HAND GESTURE AND THEN CAN RECOGNIZED THAT GESTURE
# DEVLOPER : CHANCHAL ROY

from time import time
start = time()

from mpkit_cc import Mptools
from cv2 import FONT_HERSHEY_SIMPLEX, circle,imshow, moveWindow, putText, rectangle,waitKey,destroyAllWindows
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from numpy import zeros

width, height = 1280,720
mp = Mptools(hand_no=1,win_height=height,win_width=width)
cam = mp.init()
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
WHITE = (255,255,255)
a = 1
key_points = [0,2,4,5,8,9,12,13,16,17,20]
train = True
tol = 10
known_gesture = []
train_num = 0

def get_hand_distance(hand_data):
    distance_matrix = zeros([len(hand_data),len(hand_data)],"float")

    for row in range(0,len(hand_data)):
        if hand_data != None:
            palm1 = _normalized_to_pixel_coordinates(hand_data[0][0], hand_data[0][1], width, height)
            palm2 = _normalized_to_pixel_coordinates(hand_data[9][0], hand_data[9][1], width, height)

            palm_distance = (((palm1[0] - palm2[0]) ** 2) + ((palm1[1] - palm2[1]) ** 2)) ** (1.0 / 2.0)

            for column in range(0,len(hand_data)):
                p1 = _normalized_to_pixel_coordinates(hand_data[row][0],hand_data[row][1],width,height)
                p2 = _normalized_to_pixel_coordinates(hand_data[column][0],hand_data[column][1],width,height)

                distance_matrix[row][column] = ((((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)) ** (1.0 / 2.0)) / palm_distance
    return distance_matrix

def get_error(known_gesture,unknown_gesture,key_points):
    error = 0
    
    for row in key_points:
        for column in key_points:
            error = error + abs(known_gesture[row][column] - unknown_gesture[row][column])
    return error

def find_gesture(known_gesture,unknown_gesture,key_points,gesture_names,tol):
    error_set = []

    for i in range(0,len(gesture_names),1):
        error = get_error(known_gesture[i],unknown_gesture,key_points)
        error_set.append(error)

    error_min = error_set[0]
    min_index = 0

    for i in range(0,len(error_set),1):
        if error_set[i] < error_min:
            error_min = error_set[i]
            min_index = i

    if error_min < tol:
        match = gesture_names[min_index]
    if error_min >= tol:
        match = "Unknown"

    return match

num_gesture = int(input("How many Gestures do you have: "))
print('')

gesture_names = []
for i in range(0,num_gesture,1):
    name = input(f"Please, Enter the name of your Gesture #{i + 1}: ")
    gesture_names.append(name)

print("")

while cam.isOpened():
    success,image = cam.read()
    if not success:
        print("Ignoring Empty Frame...\n")
        continue
    
    height,width,channel = image.shape
        
    hands_lm,hands_tp,hands_sr = mp.find_Hands(image,show_detect=True,hand_connection=True)

    if train == True:
        while a == 1:
            print(f"Show your Gesture {gesture_names[train_num]} and press 't' when your'e ready to train.")
            a += 1
        if hands_lm != []:
            if waitKey(1) == ord("t"):
                if hands_lm != None:
                    print(f"Training for '{gesture_names[train_num]}' Done!")
                    known_gesture.append(get_hand_distance(hands_lm[0]))
                    train_num += 1
                    a = 1
                    if train_num == num_gesture:
                        print("\nAll Trainings are done!")
                        train = False

    if train == False:
        if hands_lm != []:
            try:
                unknown_gesture = get_hand_distance(hands_lm[0])
                gesture = find_gesture(known_gesture,unknown_gesture,key_points,gesture_names,tol)

                putText(image,gesture,(200,50),FONT_HERSHEY_SIMPLEX,2,BLUE,4)

            except Exception:
                continue

    end = time()
    fps = int(1 / (end - start))
    start = end

    mp.show_FPS(image,fps_rate=fps)
    imshow("Hand Gesture : Python",image)
    moveWindow("Hand Gesture : Python",0,0)

    if waitKey(1) == ord("q"):
        break

cam.release()
destroyAllWindows()