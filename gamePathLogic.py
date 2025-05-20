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

def decode_qrs_in_row(image_path_or_img_array, num_expected_qrs=5):
    """
    Detects QR codes in a single image (expected to be a row of QRs),
    assigns them to slots, and returns a list of decoded data.

    Args:
        image_path_or_img_array: Path to the image file or a pre-loaded image (numpy array).
        num_expected_qrs: The number of QR codes expected in the row.

    Returns:
        A list of length num_expected_qrs. Each element is either:
        - A dictionary {'data': str, 'raw_orientation': ZBarOrientation enum member}
        - None if a QR code for that position is not found or decoding fails.
    """
    if isinstance(image_path_or_img_array, str):
        img = cv2.imread(image_path_or_img_array)
        if img is None:
            print(f"Error: Image not loaded from {image_path_or_img_array}")
            return [None] * num_expected_qrs
    else:
        img = image_path_or_img_array
    
    if img is None or img.size == 0:
        print(f"Error: Image is empty for QR detection.")
        return [None] * num_expected_qrs

    img_height, img_width = img.shape[:2]
    if img_width == 0: # Should not happen if img.size > 0, but defensive
        print(f"Error: Image width is zero.")
        return [None] * num_expected_qrs
        
    decoded_objects = decode(img)
    
    output_list = [None] * num_expected_qrs
    
    if not decoded_objects:
        # print("No QR codes found in the image.")
        return output_list

    detected_qrs_info = []
    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            center_x = obj.rect.left + obj.rect.width / 2.0
            detected_qrs_info.append({
                'data': obj.data.decode('utf-8'),
                'orientation': obj.orientation, # This is a ZBarOrientation enum member
                'center_x': center_x,
                'rect': obj.rect # For debugging if needed
            })

    # Sort detected QR codes by their horizontal center position.
    # This helps if multiple QRs fall into the same slot calculation,
    # we can prioritize or just be aware.
    detected_qrs_info.sort(key=lambda qr: qr['center_x'])

    slot_width = img_width / float(num_expected_qrs)
    
    for qr in detected_qrs_info:
        # Determine which slot this QR belongs to
        slot_index = int(qr['center_x'] / slot_width)
        # Ensure slot_index is within bounds [0, num_expected_qrs - 1]
        slot_index = max(0, min(slot_index, num_expected_qrs - 1))

        if output_list[slot_index] is None:
            output_list[slot_index] = {
                'data': qr['data'],
                'raw_orientation': qr['orientation'] 
            }
        else:
            # This slot is already filled. This can happen if QRs are very close,
            # slot division isn't perfect, or multiple pyzbar detections for same visual QR.
            # The current approach takes the first one (after sorting by center_x) that maps to the slot.
            print(f"Info: Slot {slot_index} was already filled. QR data '{qr['data']}' (center_x: {qr['center_x']}) "
                  f"contended for the same slot. Previous data: '{output_list[slot_index]['data']}'. Keeping first.")

    return output_list

def generate_grid_matrix_from_qr_images(row_image_paths: list, p_dict, num_qrs_per_row_image=5, grid_rows=5):
    """
    Processes multiple images, each containing a row of QR codes,
    to build a final grid matrix.

    Args:
        row_image_paths: A list of paths to the images. Each image represents one row in the grid.
        p_dict: The pathDict mapping QR data to 3x3 matrices.
        num_qrs_per_row_image: Number of QRs expected in each row image.
        grid_rows: Number of row images (and thus rows in the final QR grid).

    Returns:
        A 2D numpy array representing the assembled grid.
    """
    if not row_image_paths:
        print("Error: No image paths provided.")
        return None
        
    # Determine tile dimensions from a sample in pathDict (e.g., 'emptyPath')
    # Assuming all pathDict entries have the same matrix dimensions.
    sample_tile = p_dict.get('emptyPath', np.zeros((3,3))) # Default to 3x3 if emptyPath is missing
    tile_h, tile_w = sample_tile.shape

    assert len(row_image_paths) == grid_rows, \
        f"Expected {grid_rows} image paths for the grid rows, got {len(row_image_paths)}"

    # Initialize the final large matrix
    final_matrix_rows = grid_rows * tile_h
    final_matrix_cols = num_qrs_per_row_image * tile_w
    # Ensure dtype matches your pathDict values (usually int)
    final_grid_matrix = np.zeros((final_matrix_rows, final_matrix_cols), dtype=sample_tile.dtype)

    for grid_row_idx in range(grid_rows): # Iterates for each row image
        image_path = row_image_paths[grid_row_idx]
        print(f"\nProcessing row image {grid_row_idx + 1}/{grid_rows}: {image_path}")
        
        # decoded_qrs_in_row is a list of (QR info dict or None) for the current image
        decoded_qrs_in_row = decode_qrs_in_row(image_path, num_qrs_per_row_image)
        print(decoded_qrs_in_row)

        for grid_col_idx in range(num_qrs_per_row_image): # Iterates for QRs within the current row image
            qr_info = decoded_qrs_in_row[grid_col_idx]
            current_3x3_matrix = None

            if qr_info is None:
                # print(f"  Grid Cell ({grid_row_idx},{grid_col_idx}): No QR code found or decoding failed.")
                current_3x3_matrix = p_dict['emptyPath']
            else:
                myData = qr_info['data']
                orientation = qr_info['raw_orientation'] # pyzbar enum
                # print(f"  Grid Cell ({grid_row_idx},{grid_col_idx}): Data='{myData}', Orientation={orientation}")
                
                if myData in p_dict:
                    base_matrix_3x3 = p_dict[myData]
                    current_3x3_matrix = fixMatrixOrientation(base_matrix_3x3, orientation)
                else:
                    print(f"Warning: QR data '{myData}' not found in pathDict for grid cell ({grid_row_idx}, {grid_col_idx}). Using 'emptyPath'.")
                    current_3x3_matrix = p_dict['emptyPath']
            
            # Place the 3x3 matrix into the correct slot in the final_grid_matrix
            start_row_slice = grid_row_idx * tile_h
            end_row_slice = start_row_slice + tile_h
            start_col_slice = grid_col_idx * tile_w
            end_col_slice = start_col_slice + tile_w
            
            final_grid_matrix[start_row_slice:end_row_slice, start_col_slice:end_col_slice] = current_3x3_matrix
            
    return final_grid_matrix


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
      