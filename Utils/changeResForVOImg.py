import os
import sys
import cv2 as cv
import numpy as np
def scalebinaryimg(imgarray, goalsize):
    if not isinstance(imgarray, np.ndarray):
        return
    imheight = imgarray.shape[0]
    imwidth = imgarray.shape[1]
    ratioW =  imwidth / goalsize[1]
    ratioH = imheight / goalsize[0]
    outputarray = np.ones((goalsize[0], goalsize[1]), dtype = np.uint8) * 255
    inx = np.where(imgarray == 0)
    ii = (inx[0] / ratioH).astype(np.int)
    jj = (inx[1] / ratioW).astype(np.int)
    for i in range(len(ii)):
        if outputarray[ii[i], jj[i]] > 0:
            outputarray[ii[i], jj[i]] -= 10
            if outputarray[ii[i], jj[i]] < 0:
                outputarray[ii[i], jj[i]] = 0
    return outputarray


if __name__ == '__main__':
    assert len(sys.argv) == 4
    imgpath = sys.argv[1]
    resizeRatio = float(sys.argv[2])
    newpath = sys.argv[3]
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filelist = os.listdir(imgpath)
    inx = 1
    for file in filelist:
        newfile = str(inx) + '.png'
        inx += 1
        imgarray = cv.imread(os.path.join(imgpath, file), 0)
        sizegoal=(round(imgarray.shape[1] * resizeRatio), round(imgarray.shape[0] * resizeRatio))
        newimg = cv.resize(imgarray,sizegoal)
        outputpath = os.path.join(newpath, newfile)
        cv.imwrite(outputpath, newimg)
        if inx % 100 == 0:
            print("Processed {} files.".format(inx))
    print("Processed {} files. Over!".format(inx))