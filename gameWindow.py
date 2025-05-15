import cv2 as cv
from cv2.typing import MatLike
import re
import os

class Window:
    def __init__(self, winname: str):
        self.winname = winname

    def show(self, img: MatLike, fullscreen: bool = True, delay: int = 0):
        cv.namedWindow(self.winname, cv.WINDOW_NORMAL)
        if fullscreen:
            cv.setWindowProperty(self.winname, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        else:
            cv.setWindowProperty(self.winname, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
        cv.imshow(self.winname, img)
        cv.waitKey(delay)
        
    def close(self):
        cv.destroyWindow(self.winname)

class GameWindow(Window):
    def __init__(self, winname: str, path: str):
        super().__init__(winname)
        self.path = path
        self.images = os.listdir(path)
        self.images.sort(key=lambda x:int(re.findall(r"[0-9]+",x)[0]))  # Sort the images to ensure consistent order
        self.fullscreen = True

    def displayStage(self,stageNum:int = 1):
        assert stageNum <= len(self.images), "Stage number exceeds available images"
        img = cv.imread(os.path.join(self.path, self.images[stageNum-1]))
        self.show(img)
    
    def displayAllStages(self):
        for i in range(len(self.images)):
            img = cv.imread(os.path.join(self.path, self.images[i]))
            self.show(img)

if __name__ == "__main__":
    gameWindow = GameWindow("Game Name", r"./ui_interface")
    gameWindow.displayAllStages()
    gameWindow.close()