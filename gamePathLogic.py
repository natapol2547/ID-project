import numpy as np
import math
import cv2
from pyzbar.pyzbar import decode
'''
library for handling  qr code reading, path placement, and pathfinding
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
'''
def fixMatrixOrientation(data,direction):
    if direction == 'LEFT':
        data = np.rot90(data, k=1)
    elif direction == 'UP':
        data = np.rot90(data, k=2)
    elif direction == 'RIGHT':
        data = np.rot90(data, k=3)
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
# Define a base question template
def create_question(start_type, start_pos, obstacle_positions, end_type, end_pos):
    return {
        "start": [[start_type, start_pos]],
        "obstacle": obstacle_positions,
        "end": [[end_type, end_pos]]
    }

# Group questions by animal type
questionDict = {
    # Bear questions (1-14)
    1: create_question('bearStartPoint', 0, [1, 9, 11, 22], 'honeyEndPoint', 23),
    2: create_question('bearStartPoint', 8, [7, 13, 16], 'honeyEndPoint', 12),
    3: create_question('bearStartPoint', 10, [7, 11, 13], 'honeyEndPoint', 14),
    4: create_question('bearStartPoint', 0, [7, 13, 15, 21], 'honeyEndPoint', 24),
    5: create_question('bearStartPoint', 0, [6, 8, 16, 18], 'honeyEndPoint', 24),
    6: create_question('bearStartPoint', 9, [4, 14, 16, 22], 'honeyEndPoint', 21),
    7: create_question('bearStartPoint', 3, [8, 16], 'honeyEndPoint', 20),
    8: create_question('bearStartPoint', 9, [3, 16, 17], 'honeyEndPoint', 21),
    9: create_question('bearStartPoint', 7, [2, 11, 17, 18], 'honeyEndPoint', 22),
    10: create_question('bearStartPoint', 0, [5, 12, 16], 'honeyEndPoint', 17),
    11: create_question('bearStartPoint', 1, [2, 5, 16], 'honeyEndPoint', 21),
    12: create_question('bearStartPoint', 14, [8, 10, 16, 19], 'honeyEndPoint', 15),
    13: create_question('bearStartPoint', 23, [4, 11, 18], 'honeyEndPoint', 6),
    14: create_question('bearStartPoint', 23, [8, 10, 18, 22], 'honeyEndPoint', 2),
    
    # Nemo questions (15-19)
    15: create_question('nemoStartPoint', 11, [6, 12, 14, 19], 'coralEndPoint', 13),
    16: create_question('nemoStartPoint', 15, [10, 18], 'coralEndPoint', 4),
    17: create_question('nemoStartPoint', 1, [7, 13, 15, 20], 'coralEndPoint', 24),
    18: create_question('nemoStartPoint', 1, [7, 12, 15, 21], 'coralEndPoint', 24),
    19: create_question('nemoStartPoint', 3, [8, 12, 15, 19], 'coralEndPoint', 24),
    
    # Duck questions (20-24)
    20: create_question('duckStartPoint', 4, [6, 13, 18], 'ducklingsEndPoint', 20),
    21: create_question('duckStartPoint', 0, [2, 11, 18], 'ducklingsEndPoint', 24),
    22: create_question('duckStartPoint', 14, [16, 17, 18, 19], 'ducklingsEndPoint', 24),
    23: create_question('duckStartPoint', 0, [6, 12, 15, 17], 'ducklingsEndPoint', 24),
    24: create_question('duckStartPoint', 12, [0, 11, 16, 21], 'ducklingsEndPoint', 20),
    
    # Monkey questions (25-29)
    25: create_question('monkeyStartPoint', 1, [6, 8, 13, 15], 'bananaEndPoint', 24),
    26: create_question('monkeyStartPoint', 22, [11, 17], 'bananaEndPoint', 12),
    27: create_question('monkeyStartPoint', 9, [8, 10, 13], 'bananaEndPoint', 20),
    28: create_question('monkeyStartPoint', 0, [1, 10, 12, 19], 'bananaEndPoint', 24),
    29: create_question('monkeyStartPoint', 22, [8, 10, 17], 'bananaEndPoint', 3)
}
############################################################################

def imageToMatrix(imagePaths:dict):
    assert len(imagePaths) == 25, "Image paths must be a list of 25 images"
    QR_matrix = None
    concatenate_matrix = None
    for i in range(25):
        # img = cv2.imread(f"croppedQR_{i}.png")
        img = cv2.imread(imagePaths[i])
        decoded = decode(img)

        if len(decoded) == 0:
            temp = pathDict['emptyPath'] #unreadable qr default to empty path
        else:
            barcode = decoded[0]
            myData = barcode.data.decode('utf-8')
            print(myData, barcode.orientation)
            temp = fixMatrixOrientation(pathDict[myData], barcode.orientation)

        if concatenate_matrix is None:
            concatenate_matrix = temp
        else:
            concatenate_matrix = np.concatenate((concatenate_matrix, temp), axis=1)

        if i % 5 == 4:
            if QR_matrix is None:
                QR_matrix = concatenate_matrix
            else:
                QR_matrix = np.concatenate((QR_matrix, concatenate_matrix), axis=0)
            concatenate_matrix = None
    
    return QR_matrix

def checkAnswerCorrectBool(questionData, QR_matrix)->bool:
    if checkTilePlacement(questionData,QR_matrix) == True:
        print("Correct Placement")
        for i in range(len(questionData['start'])):
            if questionData['start'][i][0] == 'bearStartPoint':
                if Astar(2,3,1,QR_matrix) == "Not Connected":
                    return "Invalid Path"
            elif questionData['start'][i][0] == 'monkeyStartPoint':
                if Astar(4,5,1,QR_matrix) == "Not Connected":
                    return "Invalid Path"
            elif questionData['start'][i][0] == 'duckStartPoint':
                if Astar(7,8,6,QR_matrix) == "Not Connected":
                    return "Invalid Path"
            elif questionData['start'][i][0] == 'nemoStartPoint':
                if Astar(9,10,6,QR_matrix) == "Not Connected":
                    return "Invalid Path"
        return True
    else:
        return False
      