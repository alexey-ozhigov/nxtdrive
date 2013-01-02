#!/usr/bin/python

import roslib
roslib.load_manifest('cv_nxtdrive')
import rospy
import numpy as np
import cv2
from numpy import sqrt, arccos as acos, pi
from sys import stderr

def dist(p1, p2):
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    return sqrt(d1 * d1 + d2 * d2)

def test_cont(cont):
    ds = np.array([(abs(cont[i][0][0] - cont[i+1][0][0]), abs(cont[i][0][1] - cont[i+1][0][1])) for i in range(0, len(cont)-1)])
    if np.all(np.array([x[0] <= 1 and x[1] <= 1 for x in ds])):
        print 'OK'
    else:
        print np.where(np.array([x[0] > 1 or x[1] > 1 for x in ds]))

def cos_phi(p1, p2, p3):
    cross_p = (p1[0][0] - p2[0][0]) * (p3[0][0] - p2[0][0]) + (p1[0][1] - p2[0][1]) * (p3[0][1] - p2[0][1])
    norm1 = sqrt((p1[0][0] - p2[0][0]) * (p1[0][0] - p2[0][0]) + (p1[0][1] - p2[0][1]) * (p1[0][1] - p2[0][1]))
    norm2 = sqrt((p3[0][0] - p2[0][0]) * (p3[0][0] - p2[0][0]) + (p3[0][1] - p2[0][1]) * (p3[0][1] - p2[0][1]))
    try:
        ret = cross_p / (norm1 * norm2)
    except ZeroDivisionError, e:
        print e
        return -1
    return ret

def kcurv(cont, delta):
    COEF_MIN = 1.0 / (1.0 * delta)
    COEF_MAX = 1.0 * delta
    ret = []
    d = 0
    j = 0
    i = 0
    length = cv2.arcLength(cont, False)
    overlap = int(length) % delta
    k = (int(length) + overlap) / delta
    #pad contour with points from the head to be of size k*delta >= length
    cont = np.append(cont, cont[0:overlap+delta, ...], axis=0)
    #test_cont(cont)
    d_table = {(0, 1): 1.0, (1, 0): 1.0, (1, 1): sqrt(2)}
    ret.append([cont[0][0], 0])
    while True:
        while d < delta and j < len(cont) - 1:
            dx = abs(cont[j][0][0] - cont[j+1][0][0])
            dy = abs(cont[j][0][1] - cont[j+1][0][1])
            d += d_table[(dx, dy)]
            j += 1
        ret.append([cont[j][0], 0])
        d = 0
        i = j
        if j == len(cont) - 1:
            break
    #calculate k-curvature as angle between two adjacent segments
    l = len(ret)
    try:
        ret[0][1] = acos(cos_phi(ret[l-1], ret[0], ret[1]))
        for i in range(1, l-1):
            ret[i][1] = acos(cos_phi(ret[i-1], ret[i], ret[i+1]))
        ret[l-1][1] = acos(cos_phi(ret[l-2], ret[l-1], ret[0]))
    except Exception, e:
        print cos_phi(ret[l-1], ret[0], ret[1])
        print cos_phi(ret[i-1], ret[i], ret[i+1])
        print cos_phi(ret[l-2], ret[l-1], ret[0])
        rospy.signal_shutdown('acos error')
        print e
    return (ret, cont)
