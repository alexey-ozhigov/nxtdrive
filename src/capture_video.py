#!/usr/bin/python

import roslib; roslib.load_manifest('cv_nxtdrive')
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv

bridge = CvBridge()
debug = False

def capture_video_cv2():
    rospy.init_node('capture_video', anonymous=True)
    pub = rospy.Publisher('/camera/rgb/image_color', Image)
    vcap = cv2.VideoCapture(0)
    if debug:
        cv2.namedWindow('wnd')
    while True:
        (ret, img) = vcap.read()
        try:
            msg = bridge.cv_to_imgmsg(img)
            pub.publish(msg)
        except CvBridgeError, e:
            print e

        if debug:
            cv2.imshow('wnd', img)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    if debug:
        cv2.destroyAllWindows()

def capture_video_cv():
    rospy.init_node('capture_video', anonymous=True)
    pub = rospy.Publisher('/camera/rgb/image_color', Image)
    vcap = cv.CaptureFromCAM(0)
    if debug:
        cv.NamedWindow('wnd', 1)
    while not rospy.is_shutdown():
        img = cv.QueryFrame(vcap)
        try:
            msg = bridge.cv_to_imgmsg(img)
            pub.publish(msg)
        except CvBridgeError, e:
            print e

        if debug:
            cv.ShowImage('wnd', img)
            ch = 0xFF & cv.WaitKey(1)
            if ch == 27:
                break
    if debug:
        cv.DestroyAllWindows()

if __name__ == '__main__':
    capture_video_cv()
