import cv2 as cv
import re
import os
from enum import Enum
import subprocess
from playsound import playsound

UI_INTERFACE_DIR = "ui_interface"

class Window:
    def __init__(self, winname: str):
        self.winname = winname

    def show(self, img, fullscreen: bool = True, delay: int = 1):
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
    def __init__(self, winname: str, path = UI_INTERFACE_DIR, debug = False):
        super().__init__(winname)
        self.path = path
        self.soundEffectPath = r'animals effect sound'
        self.images = os.listdir(path)
        self.images.sort(key=lambda x:int(re.findall(r"[0-9]+",x)[0]))  # Sort the images to ensure consistent order
        self.fullscreen = not debug
    class soundEffects(Enum):
        START = r"game-start-317318.wav"
        BEAR =  r"Bear Sound Effect.wav"
        DUCK = r"Duck quack   Sound Effect.wav"
        FISH = r"Fish Tank Bubbles Sound Effect.wav"
        MONKEY = r"Monkey - Sound Effect.wav"
        
    def displayStage(self,stageNum:int = 1):
        assert stageNum <= len(self.images), "Stage number exceeds available images"
        img = cv.imread(os.path.join(self.path, self.images[stageNum]))
        self.show(img)
    
    def displayAllStages(self):
        for i in range(len(self.images)):
            self.displayStage(i+1)

    def playSound(self, sound: soundEffects, Dir: str = None): #call this by passing enum, example: playSound(GameWindow.soundEffects.START)s
        Dir = Dir if Dir else self.soundEffectPath
        try:
            subprocess.Popen(r"play -v 0.6 "+ '"' + os.path.join(Dir, sound.value) + '"',shell=True)
        except Exception as e:
            print(f"Error playing sound: {e}") 

if __name__ == "__main__":
    gameWindow = GameWindow("Game Name", r"./ui_interface")
    gameWindow.displayAllStages()
    gameWindow.close()
