import numpy as np
import cv2

def cropSlice(cropped:np.ndarray, sliceNum=5,pad=1/8,write=False,imageIndexOffset=0): #crops image into sliceNumSlices, each crop is padded by padded, path to write is offest by imageIndexOffset
    croppedArray = []
    imgPaths = []
    sliceNum = 4
    pad = 1/8
    for i in range(sliceNum):
        croppedImg = cropped[:,abs(int((cropped.shape[1]*i/sliceNum)-cropped.shape[1]*pad)):int((cropped.shape[1]*(i+1)/sliceNum)+cropped.shape[1]*pad)]
        croppedArray.append(croppedImg)
        if write: # write to file if true
            imgPaths.append(f"indiv_{i+imageIndexOffset+1}.png")
            cv2.imwrite(f"indiv_{i+imageIndexOffset+1}.png",croppedImg)
    return croppedArray, imgPaths


def perspTransform(image,offA=100,offB=300,shift = 350): #Apply transform to image, try to tune offA and offB, adjust shift if image falls off the left side edge
    pts1 = np.float32([[0, 0], [image.shape[1], 0],
                        [offA, offB], [image.shape[1]-offA, offB]])
    pts2 = np.float32([[0, 0], [image.shape[1], 0],
                        [0, image.shape[0]], [image.shape[1], image.shape[0]]]) + shift

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (1500, 1500))
    return result