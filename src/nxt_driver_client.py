#!/usr/bin/env python

import roslib
roslib.load_manifest('cv_nxtdrive')
import rospy
from geometry_msgs.msg import Point
import nxt.locator
from nxt.motor import *
from sys import stdout
import numpy as np

#Limits for turning angle and speed
CTL_FORWARD_VALUE_MIN = 0
CTL_FORWARD_VALUE_MAX = 100
CTL_TURN_VALUE_MIN = 0.0
CTL_TURN_VALUE_MAX = 0.6
NXT_FORWARD_VALUE_MIN = 0
NXT_FORWARD_VALUE_MAX = 100
NXT_TURN_K_MIN = 1
NXT_TURN_K_MAX = 3

NXT_CTL_TOPIC = '/nxt_motor_ctl'

UNKNOWN_ANGLE = 400
prev_ctl_turn_angle = CTL_TURN_VALUE_MIN
brick = None
l_motor = None
r_motor = None

def nxt_ctl_handler(req):
    #print 'SPEED %d TURN %d' % (req.x, req.y)
    global prev_ctl_turn_angle
    global l_motor
    global r_motor
    ctl_speed = req.x
    ctl_turn_angle = req.y
    print ctl_turn_angle
    if ctl_speed == UNKNOWN_ANGLE and ctl_turn_angle == UNKNOWN_ANGLE:
        l_motor.brake()
        r_motor.brake()
        return
    if abs(ctl_turn_angle) > CTL_TURN_VALUE_MAX:
        ctl_turn_angle = np.sign(ctl_turn_angle) * CTL_TURN_VALUE_MAX
    if ctl_turn_angle < 0:
        ctl_turn_dir = -1
    else:
        ctl_turn_dir = 1
    ctl_turn_angle = abs(ctl_turn_angle)

    ctl_forw_k = (ctl_speed - CTL_FORWARD_VALUE_MIN) / (CTL_FORWARD_VALUE_MAX - CTL_FORWARD_VALUE_MIN)
    nxt_speed = NXT_FORWARD_VALUE_MIN + ctl_forw_k * (NXT_FORWARD_VALUE_MAX - NXT_FORWARD_VALUE_MIN)
    nxt_forw_k = (nxt_speed - NXT_FORWARD_VALUE_MIN) / \
                 (NXT_FORWARD_VALUE_MAX - NXT_FORWARD_VALUE_MIN)
    if nxt_forw_k < 0.1:
        nxt_speed = 0
    #nxt_speed = 10
    ctl_turn_k = (ctl_turn_angle - CTL_TURN_VALUE_MIN) / (CTL_TURN_VALUE_MAX - CTL_TURN_VALUE_MIN)
    nxt_turn_k = NXT_TURN_K_MIN + ctl_turn_k * (NXT_TURN_K_MAX - NXT_TURN_K_MIN)
    if ctl_turn_dir < 0:
        l_speed = nxt_speed
        r_speed = int(nxt_turn_k * nxt_speed)
        if r_speed < -128:
            r_speed = -128
        if r_speed > 127:
            r_speed = 127
    else:
        r_speed = nxt_speed
        l_speed = int(nxt_turn_k * nxt_speed)
        if l_speed < -128:
            l_speed = -128
        if l_speed > 127:
            l_speed = 127
    l_motor.run(l_speed, regulated = True)   
    r_motor.run(r_speed, regulated = True)   

def run_nxt_driver_client():
    global brick
    global l_motor
    global r_motor
    brick = nxt.locator.find_one_brick(name = 'NXT')
    l_motor = Motor(brick, PORT_C)
    r_motor = Motor(brick, PORT_B)
    rospy.init_node('nxt_bluetooth_driver_wrapper')
    ctl_sub = rospy.Subscriber(NXT_CTL_TOPIC, Point, nxt_ctl_handler)
    print 'SEARCHING FOR NXT BLUETOOTH ... ',
    stdout.flush()
    print 'DONE'
    rospy.spin()

if __name__ == '__main__':
    run_nxt_driver_client()
