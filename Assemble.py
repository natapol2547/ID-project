# from pyzbar.pyzbar import decode
import numpy as np
import math
import cv2
import random
import Jetson.GPIO as GPIO
import time

#########area for QR code cropping def###########








###############################################

def fixMatrixOrientation(data,direction):
    if direction == 'LEFT':
        data = np.rot90(data, k=3)
    elif direction == 'DOWN':
        data = np.rot90(data, k=2)
    elif direction == 'RIGHT':
        data = np.rot90(data, k=1)
    return data

################code for checking if the initial tile placement is correct according to the question##############
def observe(matrix_value,pos,matrix):
    temp  = matrix[math.floor(pos/5)*3:math.floor(pos/5)*3+3,pos%5*3:pos%5*3+3]
    if False in (matrix_value == temp):
        return False
    else:
        return True

def checkTilePlacement(question,matrix):
    for i in range(len(question['start'])):
        matrix_value = pathDict[question['start'][i][0]]
        if observe(matrix_value,question['start'][i][1],matrix) == False:
            return False
        
    for i in range(len(question['obstacle'])):
        if observe(pathDict['obstacle'],question['obstacle'][i],matrix) == False:
            return False

    for i in range(len(question['end'])):
        matrix_value = pathDict[question['end'][i][0]]
        if observe(matrix_value,question['end'][i][1],matrix) == False:
            return False    
    return True 

############ Most ineffiecient A* #####################
def matrix_dis(current_position, destination, matrix):
    a = np.where(matrix == destination)
    diff = current_position - np.array([a[0][0], a[1][0]])
    return math.sqrt(diff[0]**2 + diff[1]**2)

def makeCost(destination, matrix):
    cost = np.empty(np.shape(matrix))
    for i in range(np.shape(cost)[0]):
        for j in range(np.shape(cost)[1]):
            cost[i][j] = matrix_dis(np.array([i, j]),destination, matrix)
    return cost

def makeCheckedPathMatrix(list, matrix):
    path = np.empty(np.shape(matrix))
    for i in range(np.shape(path)[0]):
        for j in range(np.shape(path)[1]):
            if matrix[i][j] in list:
                path[i][j] = 0
            else:
                path[i][j] = 1
    return path

def searchNode(current_pos,matrix):
    searched = np.array([[current_pos[0],current_pos[1]+1],[current_pos[0]+1,current_pos[1]],[current_pos[0],current_pos[1]-1],[current_pos[0]-1,current_pos[1]]])
    deleteList = []
    
    for i in range(3,-1,-1):
        isInvalid = False
        for j in range(1,-1,-1):
            if searched[i][j] <= 0 or searched[i][j] >= np.shape(matrix)[0] or searched[i][j] >= np.shape(matrix)[1]:
                isInvalid = True
        if isInvalid:
            deleteList.append(i)

    for i in range(len(deleteList)):
        searched = np.delete(searched, deleteList[i],axis=0)     
  
    valueInSearched = np.array([[0]])
    for i in range(len(searched)):
        value = matrix[searched[i][0]][searched[i][1]]
        valueInSearched = np.append(valueInSearched,np.array([[value]]),axis=0)
    valueInSearched = np.delete(valueInSearched,0,axis=0)
    valueInSearched = np.transpose(valueInSearched)

    return searched, valueInSearched

def Astar(start,end,path,matrix):
    G_cost = makeCost(start,matrix)
    H_cost = makeCost(end,matrix)
    F_cost = G_cost + H_cost
    Inf = 100 #any value high af so code just skip this
    
    checked_path = makeCheckedPathMatrix([start,end,path],matrix)

    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if checked_path[i][j] == 1:
                F_cost[i][j] = Inf

    path_count = (matrix == path).sum()

    #start algorithm nigga
    start_loc = np.transpose(np.where(matrix==start))[0]

    checked_path[np.where(matrix==start)] = 1
    F_cost_searched = np.full(shape=np.shape(matrix), fill_value=Inf)
    a,b = searchNode(start_loc,matrix)
    value_searched = np.array([])
    value_searched = np.concatenate((value_searched,b[0]))

    for i in range(len(a)):
        F_cost_searched[a[i][0]][a[i][1]] = F_cost[a[i][0]][a[i][1]]

    while path_count != 0:
        searchAt = np.transpose(np.where(F_cost_searched==np.min(F_cost_searched)))
        a,b = searchNode(searchAt[0],matrix)
        checked_path[searchAt[0][0],searchAt[0][1]] = 1
        
        for i in range(len(a)):
            if checked_path[a[i][0]][a[i][1]] != 1:
                F_cost_searched[a[i][0]][a[i][1]] = F_cost[a[i][0]][a[i][1]]
            else:
                F_cost_searched[a[i][0]][a[i][1]] = Inf
            if matrix[a[i][0]][a[i][1]] != path:
                    F_cost_searched[a[i][0]][a[i][1]] = Inf
        F_cost_searched[searchAt[0][0],searchAt[0][1]] = Inf 

        value_searched = np.concatenate((value_searched,b[0]))
        path_count -= 1
        # turn on the code below for some stupid ass visual
        # print(F_cost_searched)
        if end in value_searched:
            return "Connected"
    
    return "Not Connected"
###############################################################
pathDict = {
    'landStraightPath': np.array([[0,0,0],[1,1,1],[0,0,0]]),
    'landCurvePath' : np.array([[0,0,0],[0,1,1],[0,1,0]]),
    'waterStraightPath' : np.array([[0,0,0],[6,6,6],[0,0,0]]),
    'waterCurvePath' : np.array([[0,0,0],[0,6,6],[0,6,0]]),
    'bearStartPoint' : np.array([[0,1,0],[1,2,1],[0,1,0]]),
    'monkeyStartPoint' : np.array([[0,1,0],[1,4,1],[0,1,0]]),
    'duckStartPoint' : np.array([[0,6,0],[6,7,6],[0,6,0]]),
    'nemoStartPoint' : np.array([[0,6,0],[6,9,6],[0,6,0]]),
    'honeyEndPoint' : np.array([[0,1,0],[1,3,1],[0,1,0]]),
    'bananaEndPoint' : np.array([[0,1,0],[1,5,1],[0,1,0]]),
    'ducklingsEndPoint' : np.array([[0,6,0],[6,8,6],[0,6,0]]),
    'coralEndPoint' : np.array([[0,6,0],[6,10,6],[0,6,0]]),
    'obstacle' : np.array([[11,11,11],[11,11,11],[11,11,11]]),
    'emptyPath': np.array([[0,0,0],[0,0,0],[0,0,0]]),

    #temporaray path dict for testing
    # 'straightPath' : np.array([[0,0,0],[1,1,1],[0,0,0]]),
    # 'curvePath' : np.array([[0,0,0],[0,1,1],[0,1,0]]),
    # 'tPath' : np.array([[0,0,0],[1,1,1],[0,1,0]]),
    # 'plusPath' : np.array([[0,1,0],[1,1,1],[0,1,0]]),
    # 'startPoint': np.array([[0,1,0],[1,2,1],[0,1,0]]),
    # 'endPoint' : np.array([[0,1,0],[1,3,1],[0,1,0]])
}

#question dict area#
question_1 = {
    "start": [['bearStartPoint',0]],
    "obstacle" : [1,9,11,22],
    "end":[['honeyEndPoint',23]]
}
question_2 = {
    "start": [['bearStartPoint',8]],
    "obstacle" : [7,13,16],
    "end":[['honeyEndPoint',12]]
}
question_3 = {
    "start": [['bearStartPoint',10]],
    "obstacle" : [7,11,13],
    "end":[['honeyEndPoint',14]]
}
question_4 = {
    "start": [['bearStartPoint',0]],
    "obstacle" : [7,13,15,21],
    "end":[['honeyEndPoint',24]]
}
question_5 = {
    "start": [['bearStartPoint',0]],
    "obstacle" : [6,8,16,18],
    "end":[['honeyEndPoint',24]]
}
question_6 = {
    "start": [['bearStartPoint',9]],
    "obstacle" : [4,14,16,22],
    "end":[['honeyEndPoint',21]]
}
question_7 = {
    "start": [['bearStartPoint',3]],
    "obstacle" : [8,16],
    "end":[['honeyEndPoint',20]]
}
question_8 = {
    "start": [['bearStartPoint',9]],
    "obstacle" : [3,16,17],
    "end":[['honeyEndPoint',21]]
}
question_9 = {
    "start": [['bearStartPoint',7]],
    "obstacle" : [2,11,17,18],
    "end":[['honeyEndPoint',22]]
}
question_10 = {
    "start": [['bearStartPoint',0]],
    "obstacle" : [5,12,16],
    "end":[['honeyEndPoint',17]]
}
question_11 = {
    "start": [['bearStartPoint',1]],
    "obstacle" : [2,5,16],
    "end":[['honeyEndPoint',21]]
}
question_12 = {
    "start": [['bearStartPoint',14]],
    "obstacle" : [8,10,16,19],
    "end":[['honeyEndPoint',15]]
}
question_13 = {
    "start": [['bearStartPoint',23]],
    "obstacle" : [4,11,18],
    "end":[['honeyEndPoint',6]]
}
question_14 = {
    "start": [['bearStartPoint',23]],
    "obstacle" : [8,10,18,22],
    "end":[['honeyEndPoint',2]]
}
question_15 = {
    "start": [['bearStartPoint',23]],
    "obstacle" : [4,11,18],
    "end":[['honeyEndPoint',6]]
}
question_16 = {
    "start": [['nemoStartPoint',15]],
    "obstacle" : [10,18],
    "end":[['coralEndPoint',4]]
}
question_17 = {
    "start": [['nemoStartPoint',11]],
    "obstacle" : [6,12,1,19],
    "end":[['coralEndPoint',13]]
}
question_18 = {
    "start": [['nemoStartPoint',2]],
    "obstacle" : [7,13,15,20],
    "end":[['coralEndPoint',24]]
}
question_19 = {
    "start": [['nemoStartPoint',15]],
    "obstacle" : [7,12,15,21],
    "end":[['coralEndPoint',24]]
}
question_20 = {
    "start": [['nemoStartPoint',3]],
    "obstacle" : [8,12,15,19],
    "end":[['coralEndPoint',24]]
}
question_21 = {
    "start": [['monkeyStartPoint',1]],
    "obstacle" : [6,8,13,15],
    "end":[['bananaEndPoint',24]]
}
question_22 = {
    "start": [['monkeyStartPoint',1]],
    "obstacle" : [6,8,13,15],
    "end":[['bananaEndPoint',24]]
}
question_100 = {
    "start": [['bearStartPoint',12]],
    "obstacle" : [8,13,18,23],
    "end":[['honeyEndPoint',24]]
}
questionDict = {1:question_1,
                2:question_2
                }
############################################################################
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,     
    capture_height=2464,
    display_width=1920, #102
    display_height=1080,#77
    framerate=20,           # You can try higher FPS if needed
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def take_images():
    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    start_time = time.time()
    image_1_saved = False
    image_2_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        current_time = time.time()
        elapsed = current_time - start_time

        if not image_1_saved and elapsed >= 0.5:
            cv2.imwrite("image_1.png", frame)
            print("Saved image_1.png at", elapsed, "seconds")
            image_1_saved = True

   
        if not image_2_saved and elapsed >= 2:
            cv2.imwrite("image_2.png", frame)
            print("Saved image_2.png at", elapsed, "seconds")
            image_2_saved = True
            break  
            
############################################################################
def main():
    #At start: 
    QUESTION_BUT_PIN = 24
    ANSWER_BUT_PIN = 18
    SERVO_PIN = 16
    LED_PIN = 12 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(QUESTION_BUT_PIN, GPIO.IN)
    GPIO.setup(ANSWER_BUT_PIN, GPIO.IN) 
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT) 
    GPIO.add_event_detect(QUESTION_BUT_PIN, GPIO.RISING, bouncetime=200)
    GPIO.add_event_detect(ANSWER_BUT_PIN, GPIO.RISING, bouncetime=200)

    randomNumber = random.randint(1,2)
    randomQuestion  = questionDict[randomNumber]

    GPIO.output(LED_PIN, GPIO.HIGH)
    GPIO.output(SERVO_PIN,GPIO.LOW)
    #send question to display screen

    global cap
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
  
    while True:
    
        if GPIO.event_detected(QUESTION_BUT_PIN):
            print('change question')
            randomNumber = random.randint(1,2)
            randomQuestion  = questionDict[randomNumber]
            # send new question to display screen

        if GPIO.event_detected(ANSWER_BUT_PIN):
            print('check answer')
            GPIO.output(LED_PIN, GPIO.LOW)

            GPIO.output(SERVO_PIN,GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(SERVO_PIN, GPIO.LOW)
            
            take_images()
            time.sleep(0.7)
            GPIO.output(LED_PIN, GPIO.HIGH)

            #camera take pic and crop image into 25 qr images
            # QR_matrix = None
            # concatenate_matrix = None
            # for i in range(25):
            #     img = cv2.imread(f"croppedQR_{i}.png")
            #     decoded = decode(img)

            #     if len(decoded) == 0:
            #         temp = pathDict['emptyPath']
            #     else:
            #         barcode = decoded[0]
            #         myData = barcode.data.decode('utf-8')
            #         temp = fixMatrixOrientation(pathDict[myData], barcode.orientation)

            #     if concatenate_matrix is None:
            #         concatenate_matrix = temp
            #     else:
            #         concatenate_matrix = np.concatenate((concatenate_matrix, temp), axis=1)

            #     if i % 5 == 4:
            #         if QR_matrix is None:
            #             QR_matrix = concatenate_matrix
            #         else:
            #             QR_matrix = np.concatenate((QR_matrix, concatenate_matrix), axis=0)
            #         concatenate_matrix = None

            #     print(QR_matrix)
            #     # check correct placement and validness of path       
            #     tempQuestion = questionDict[randomQuestion]
            #     if checkTilePlacement(tempQuestion,QR_matrix) == True:
            #         print("Correct Placement")
            #         for i in range(len(tempQuestion['start'])):
            #             if tempQuestion['start'][i][0] == 'bearStartPoint':
            #                 if Astar(2,3,1,QR_matrix) == "Not Connected":
            #                     return "Incorrect Path"
            #             elif tempQuestion['start'][i][0] == 'monkeyStartPoint':
            #                 if Astar(4,5,1,QR_matrix) == "Not Connected":
            #                     return "Incorrect Path"
            #             elif tempQuestion['start'][i][0] == 'duckStartPoint':
            #                 if Astar(7,8,6,QR_matrix) == "Not Connected":
            #                     return "Incorrect Path"
            #             elif tempQuestion['start'][i][0] == 'nemoStartPoint':
            #                 if Astar(9,10,6,QR_matrix) == "Not Connected":
            #                     return "Incorrect Path"
            #         return "Correct Pathing"
            #     else:
            #         return "Incorrect Path Placement"
    
        time.sleep(0.1)

if __name__ == '__main__':
    main()



    #check answer button pressed:
    


    # QR_matrix set for debugging
#     QR_matrix = np.array([
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 1, 2, 1, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 1, 3, 1],
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 11, 11, 0, 1, 0] ])
    



#Land path ID: 1
#bear ID: 2
#honey ID: 3
#monkey ID: 4
#banana ID: 5

#Water path ID: 6
#Duck ID: 7
#Duckling ID: 8
#Nemo ID: 9
#Coral ID: 10

#Obstacles ID: 11