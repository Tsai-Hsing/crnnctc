from funcOD import *
from PIL import Image
import sys

try:

    resultData = mainPreLoadModel('./testfolder/')
    #mainPredict('','./testfolder/', '' ,xx['result'], '', '')
    im = Image.open('./testfolder/002le.png')
    mainPredict(im, './testfolder/' , '', resultData['result'], '', '')
    

    #res = test_detector.test(im, './testfolder/','','')
    #print(res)

except:
    print("Unexpected error:", sys.exc_info()[0])

