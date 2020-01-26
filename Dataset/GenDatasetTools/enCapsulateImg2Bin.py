#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/04
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : enCapsulateImg2Bin.py

"""
Function:
    Preprocess all the image:
        write to binary for efficiency in data loading process.
Usage:
    python enCapsulateImg2Bin.py imgDir [inxfile]
"""

import torch
import cv2 as cv
import io
import os
import sys
if __name__ == '__main__':
    assert len(sys.argv) > 1
    imgfolder = sys.argv[1]
    inxlist = None
    if len(sys.argv) > 2:
        imgInxFile = sys.argv[2]
        inxlist = open(imgInxFile,'r').readlines()
        inxlist = [str(int((i.split(',')[-1][:-1]))) for i in inxlist]

    imgfilelist = os.listdir(imgfolder)
    imgfilelist = [i for i in imgfilelist if i.split('.')[-1] == 'png']
    imgfilelist.sort(key=lambda x:int(x[:-4]))
    imgbuffersize = None
    imgbinfile = os.path.join(imgfolder, "img.tbin")
    imgbininfofile = os.path.join(imgfolder, "imgbininfo.txt")
    inx = 0

    # Write img tensor to the *.tbin file.
    with open(imgbinfile, 'wb') as fout:
        if inxlist:
            for file in inxlist:
                inx += 1
                buffer = io.BytesIO()
                filename = os.path.join(imgfolder, file + '.png')
                img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
                img = torch.tensor(img)
                torch.save(img, buffer)
                if imgbuffersize == None:
                    imgbuffersize = len(buffer.getbuffer())
                buf = buffer.getbuffer()
                # Check image size
                assert len(buf) == imgbuffersize
                fout.write(buf)
                if inx % 100 == 0:
                    print("Processed {} images.".format(inx))
        else:
            for file in imgfilelist:
                if file[-4:] == '.png':
                    inx += 1
                    buffer = io.BytesIO()
                    filename = os.path.join(imgfolder, file)
                    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
                    img = torch.tensor(img)
                    torch.save(img, buffer)
                    if imgbuffersize == None:
                        imgbuffersize = len(buffer.getbuffer())
                    buf = buffer.getbuffer()
                    # Check image size
                    assert len(buf) == imgbuffersize
                    fout.write(buf)
                    if inx % 100 == 0:
                        print("Processed {} images.".format(inx))
    print("Processed {} images. Finished.\n".format(inx))

    # Write some information for the *.tbin file.
    with open(imgbininfofile, 'w') as fout:
        fout.write("imgBufferSize:{}\n".format(imgbuffersize))
        fout.write("imgNum:{}".format(inx))

