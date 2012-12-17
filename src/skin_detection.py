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
from myutils import calc_back_proj, hsv_filter_mask
from histogram import Histogram
from os.path import realpath

skin_hist_file  = 'skin_histogram.txt'
DEF_TOPIC_NAME = '/camera/rgb/image_color'
DEF_HISTOGRAM_FILE = '/home/alex/MAS/CV/project/cv_nxtdrive/hand_hist.hst'
bridge = CvBridge()

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

class SkinDetector:
    def __init__(self, topic = DEF_TOPIC_NAME, hist_file = DEF_HISTOGRAM_FILE):
        rospy.init_node('skin_detection', anonymous=True)
        rospy.Subscriber(topic, Image, self.rgb_cb)
        cv2.namedWindow('wnd_orig')
        cv2.namedWindow('wnd_prob')
        cv2.namedWindow('wnd_skin')
        self.skin_threshold = 127
        cv2.createTrackbar('track_skin', 'wnd_skin', self.skin_threshold, 255, self.on_skin_track)
        rospy.on_shutdown(self.on_shutdown)
        self.hist = Histogram()
        self.hist.load(hist_file)
 
    def on_skin_track(self, pos, *argv):
        self.skin_threshold = pos

    def find_skin(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = hsv_filter_mask(img_hsv)
        back_proj = calc_back_proj(img_hsv, self.hist.hist, hsv = True)
        back_proj &= mask
        skin_bin = cv2.threshold(back_proj, self.skin_threshold, 255, cv2.THRESH_BINARY)
        return (back_proj, skin_bin[1])

    def rgb_cb(self, msg):
        try:
            img = bridge.imgmsg_to_cv(msg, 'bgr8')
            img = np.asarray(img)
        except CvBridgeError, e:
            print >>stderr, 'Cannot convert from ROS msg to CV image:', e

        (img_prob, img_skin) = self.find_skin(img)
        cv2.imshow('wnd_orig', img)
        cv2.imshow('wnd_prob', img_prob)
        cv2.imshow('wnd_skin', img_skin)
        ch = cv2.waitKey(3)
        if ch == 27:
            rospy.signal_shutdown('Quit')
        elif ch == ord(' '):
            cv2.imwrite('img_prob.png', img_prob)

    def run(self):
        rospy.spin()
 
    def on_shutdown(self):
        cv2.destroyAllWindows()
        cv2.waitKey(100)

def skin_detector(topic_name, hist_file):
    skdet = SkinDetector(topic_name, hist_file)
    skdet.run()

if __name__ == '__main__':
    skin_detector(DEF_TOPIC_NAME, DEF_HISTOGRAM_FILE)
