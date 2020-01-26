# It is disigned for single-line laser file: .lms, maxRange=100m
import struct
import numpy as np
import math
import matplotlib.pyplot as plt
import os

class laserscandata:
    def __init__(self):
        self.pointnum = 0
        self.anglelist = []
        self.distlist = []
        self.pointlist = []
        self.timestamp = 0
        self.globalpt = []

    def pointprojection(self, pos, skip=1, rangeMax = 80.0):
        self.globalpt = []
        offset = 0*math.pi/2
        sin_ = math.sin(pos[2] + offset)
        cos_ = math.cos(pos[2] + offset)
        trans = np.array([[cos_, sin_], [-sin_, cos_]])
        # idx = np.array([i for i in range(0, self.pointnum, skip) if self.distlist[i] < 80])
        idx_skipped  = np.arange(0, self.pointnum, skip)
        valid = np.where(self.distlist[idx_skipped] < rangeMax)
        validinx = idx_skipped[valid]
        self.globalpt = np.repeat(np.array([pos[0:2]]), len(validinx), axis=0) + np.dot(self.pointlist[validinx, :], trans)

class lmswriter():
    def __init__(self, fileName, anglerange, angleres, disunit):
        self.filename = fileName
        self.filestream = open(self.filename, 'wb')
        params = struct.pack('fff', anglerange, angleres, disunit)
        self.filestream.write(params)
        self.disunit = disunit

    def writedata(self, data, timeformat = None, reverse = False):
        if not isinstance(data, laserscandata):
            return -1
        if timeformat == 'long long':
            t = struct.pack('q', data.timestamp)
        else:
            t = struct.pack('i', data.timestamp)
        self.filestream.write(t)
        if not reverse:
            rangesbuf = b''.join(struct.pack('h', int(elem * self.disunit)) for elem in data.distlist)
        else:
            rangesbuf = b''.join(struct.pack('h', int(data.distlist[len(data.distlist) - i - 1] * self.disunit)) for i in range(len(data.distlist)))
        self.filestream.write(rangesbuf)

    def close(self):
        self.filestream.close()

class lmsgenerator():
    def __init__(self, fileName, calibpara = [0.0,0.0,0.0]):
        self.filename = fileName
        self.isend = False
        self.isopen = False
        self.currentdta = laserscandata()
        self.filestream = 0
        # self.calib = np.array([0.72, 2.73, 0.165])
        self.calib = np.array(calibpara)
        self.outputframeinx = 0


    def open(self, reverse):
        self.filestream = open(self.filename, 'rb')
        self.isopen = True
        x = self.filestream.read(12)
        params = struct.unpack('fff', x)
        self.anglerange = params[0]
        self.angleres = params[1]
        self.disunit = params[2]
        self.pointnum = int(self.anglerange / self.angleres) + 1
        self.calframe()
        self.reverse = reverse
        self.angleLRbias = (self.anglerange - 180) / 2
        return self.isopen

    def getdata(self):
        currrentdta = laserscandata()
        x = self.filestream.read(4)
        if x == b'':
            return -1
        currrentdta.timestamp = struct.unpack('i', x)[0]
        x = self.filestream.read(2 * self.pointnum)
        if x == b'':
            return -1
        currrentdta.distlist = struct.unpack('h' * self.pointnum, x)
        currrentdta.distlist = np.array(currrentdta.distlist, dtype=float) / self.disunit
        currrentdta.pointnum = self.pointnum
        currrentdta.pointlist = np.zeros([currrentdta.pointnum, 2])
        for i in range(currrentdta.pointnum):
            theta = self.angleres * i
            if self.reverse:
                currrentdta.pointlist[i, 0] = self.calib[0] + currrentdta.distlist[i] * math.cos(math.pi - math.radians(theta) + self.calib[2])
                currrentdta.pointlist[i, 1] = self.calib[1] + currrentdta.distlist[i] * math.sin(math.pi - math.radians(theta) + self.calib[2])
            else:
                currrentdta.pointlist[i, 0] = self.calib[0] + currrentdta.distlist[i] * math.cos(math.radians(theta) + self.calib[2])
                currrentdta.pointlist[i, 1] = self.calib[1] + currrentdta.distlist[i] * math.sin(math.radians(theta) + self.calib[2])
        self.outputframeinx += 1
        return currrentdta

    def calframe(self):
        self.headbytesnum = 12
        self.framebytesnum = 4 + 2 * self.pointnum
        self.totalbytes = os.path.getsize(self.filename)
        self.totalframenum = int((self.totalbytes - self.headbytesnum)/self.framebytesnum)

    def jumptoframe(self, framenum):
        if framenum > self.totalframenum:
            return -1
        self.filestream.seek(self.headbytesnum + self.framebytesnum * (framenum - 1), os.SEEK_SET)

    def close(self):
        self.filestream.close()

def vislaserdata(data):
    fig = plt.plot(data.pointlist[:,0],data.pointlist[:,1],'.')
    plt.show()

if __name__ == '__main__':
    import time
    lms = lmsgenerator('/home/akira/Project/build-gettraj-Desktop-Debug/fakelms.lms')
    lms.open()
    data = lms.getdata()
    data = lms.getdata()

    while(True):
        data = lms.getdata()
        if data == -1:
            break
        plt.plot(data.pointlist[:, 0], data.pointlist[:, 1], '.')
        plt.axis('equal')
        plt.show()
        time.sleep(0.1)



