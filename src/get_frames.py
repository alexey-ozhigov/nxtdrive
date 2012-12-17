#!/usr/bin/env python
import roslib; roslib.load_manifest('cv_nxtdrive')
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from os import errno
from os.path import basename 
from sys import exit
from sys import argv
from sys import stderr
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

bag_fname  = ''
frames_dir = ''

'''
def talker():
    pub = rospy.Publisher('chatter', String)
    while not rospy.is_shutdown():
        str = "hello world %s"%rospy.get_time()
        rospy.loginfo(str)
        pub.publish(String(str))
        rospy.sleep(1.0)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
'''

CONDITION_OCC1 = 1
CONDITION_OCC2 = 2
CONDITION_OCC3 = 3

bridge = CvBridge()
wnd = cv2.namedWindow('wnd')

class OcclusionDetector:
    cond = CONDITION_OCC1
    def __init__(self, cond):
        pass
    def feedFrame(self, frame):
        pass


def depth_image_callback(data):
    try:
        img = bridge.imgmsg_to_cv(data)
        img = np.asarray(img)
        #cv2.imshow('wnd', img) 
    except CvBridgeError, e:
        print e
    #cv2.waitKey(3)
    print 'Total NaNs: %d (320, 240) %d' % (np.isnan(img).sum(), img[240][320])
    #rospy.loginfo(rospy.get_name() + 'I heard %s\n', data.header)

def camera_info_callback(data):
    pass
    #rospy.loginfo(rospy.get_name() + 'I heard %s', data.header)

def color_image_callback(data):
    try:
        img = bridge.imgmsg_to_cv(data, 'bgr8')
        img = np.asarray(img)
        fname = frames_dir + '/frame_%d.jpg' % data.header.seq
        cv2.imwrite(fname, img)
        cv2.imshow('wnd', img) 
        cv2.waitKey(3)
        print 'writing', basename(fname)
    except CvBridgeError, e:
        print e

def listener(topic_name):
    rospy.init_node('cv_get_frames', anonymous=True)
    #rospy.Subscriber('/camera/depth_registered/image_raw', Image, depth_image_callback)
    rospy.Subscriber(topic_name, Image, color_image_callback)
    #rospy.Subscriber('/camera/rgb/camera_info', Image, camera_info_callback)
    rospy.spin()


if __name__ == '__main__':
    if len(argv) < 3:
        print 'USAGE: get_frames <bag_file> <target_dir_home> <topic_name>'
        exit(1)
    bag_fname = argv[1]
    r = re.compile('\.')
    frames_dir = re.sub(r, '_', bag_fname) + '_frames'
    try:
        os.mkdir(frames_dir)
        os.chdir(frames_dir)
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            print >>stderr, 'Cannot create directory %s, errno %d', \
                            (frames_dir, e.errno)
            exit(1)
    topic_name = argv[2]
    listener(topic_name)
