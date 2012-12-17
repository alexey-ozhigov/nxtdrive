#!/usr/bin/python

import roslib
roslib.load_manifest('cv_nxtdrive')
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from cv2.cv import RGB
import numpy as np
import os
from os import errno
from os.path import basename
from sys import exit
from sys import argv
from sys import stderr
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import sleep
from math import sqrt
import pickle
depth_hist_file = 'depth_histogram.txt'
DEF_TOPIC_NAME = '/camera/depth_registered/image_rect'
bridge = CvBridge()

class Enum(set):

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


ThresholdMode = Enum(['DEPTH_AUTO', 'DEPTH_MANUAL'])

def dist(p1, p2):
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    return sqrt(d1 * d1 + d2 * d2)


def build_line(p1, p2, debug_img = False):
    ret_line = np.empty([0, 1, 2], dtype='int32')
    p_cur = [p1[0], p1[1]]
    x_delta = 1 if p1[0] < p2[0] else -1
    y_delta = 1 if p1[1] < p2[1] else -1
    delta = (x_delta, y_delta)
    w = abs(p1[0] - p2[0])
    h = abs(p1[1] - p2[1])
    acc_delta = (w, h)
    acc = [w, h]
    cur_acc, other_acc = (1, 0) if acc[0] > acc[1] else (0, 1)
    i = 0
    while np.any(p_cur != p2):
        ret_line = np.append(ret_line, np.array([[[p_cur[0], p_cur[1]]]], dtype='int32'), axis=0)
        if acc[cur_acc] == acc[other_acc]:
            p_cur[cur_acc] += delta[cur_acc]
            p_cur[other_acc] += delta[other_acc]
            acc[cur_acc] += acc_delta[cur_acc]
            cur_acc, other_acc = other_acc, cur_acc
        elif acc[cur_acc] < acc[other_acc]:
            p_cur[other_acc] += delta[other_acc]
        elif acc[cur_acc] > acc[other_acc]:
            p_cur[cur_acc] += delta[cur_acc]
            cur_acc, other_acc = other_acc, cur_acc
        acc[cur_acc] += acc_delta[cur_acc]

    if debug_img:
        img = np.zeros([max(p1[1], p2[1]) + 1, max(p1[0], p2[0]) + 1], dtype='uint8')
        for p in ret_line:
            _p = p[0]
            img[_p[1]][_p[0]] = 255

    else:
        img = None
    return (ret_line, img)


def my_find_contours(img):
    visited = np.ones(img.shape, dtype='bool')


def max_filter(img):
    s = img.shape
    img_f = np.zeros((s[0] + 2, s[1] + 2), dtype='uint8')
    img_ret = np.zeros((s[0], s[1]), dtype='uint8')
    img_f[1:s[0] + 1, 1:s[1] + 1] = img
    for i in range(1, s[0]):
        for j in range(1, s[1]):
            img_ret[i][j] = np.max(img_f[i - 1:i + 2, j - 1:j + 2])

    return img_ret


class DepthDetector():
    MIN_DIST = 3.0

    def __init__(self, topic = DEF_TOPIC_NAME, mode = ThresholdMode.DEPTH_AUTO):
        self.mouse_x = 0
        self.mouse_y = 0
        self.MAX_RANGE = 5.0
        self.write_dist_hist = True
        self.dist_hist_sample_size = 100
        self.dist_hist_sample_cnt = 0
        self.dist_hist_sample_number = 0
        self.stat_pixel_pos = (175, 282)
        self.dist_hist_fname_base = 'dist_hist'
        self.dist_pixel_fname = 'dist_pixel.txt'
        self.dist_hist_imgs = np.zeros([self.dist_hist_sample_size, 480, 640])
        self.img_filter_buf = self.MAX_RANGE * np.ones([480, 640])
        self.new_img_filter_buf = self.MAX_RANGE * np.ones([480, 640])
        self.dist_hist_imgs_i = 0
        self.debug_cnt = 0
        self.mode = mode
        self.depth_frame_cnt = 0
        rospy.init_node('track_hand', anonymous=True)
        cv2.namedWindow('wnd_orig', 0)
        cv2.namedWindow('wnd_prob', 0)
        cv2.namedWindow('wnd_contours', 0)
        cv2.namedWindow('wnd_debug', 0)
        cv2.setMouseCallback('wnd_orig', self.on_orig_mouse, None)
        self.depth_threshold = 127
        if self.mode == ThresholdMode.DEPTH_MANUAL:
            cv2.createTrackbar('track_depth', 'wnd_prob', self.depth_threshold, 255, self.on_depth_track)
        rospy.on_shutdown(self.on_shutdown)
        rospy.Subscriber(topic, Image, self.depth_cb)
        rospy.Subscriber('/camera/rgb/image_rect_color', Image, self.rgb_cb)

    def on_orig_mouse(self, event, x, y, i, data):
        self.mouse_x = x
        self.mouse_y = y

    def on_depth_track(self, pos, *argv):
        self.depth_threshold = pos

    def get_auto_threshold(self, img):
        min_dist_m = np.min(img)
        print 'MIN_DIST', min_dist_m
        return min_dist_m

    def get_foreground(self, img):
        RANGE_DELTA_M = 0.15
        if self.mode == ThresholdMode.DEPTH_MANUAL:
            d_thr = self.depth_threshold / 256.0 + 0.5
        elif self.mode == ThresholdMode.DEPTH_AUTO:
            d_thr = self.get_auto_threshold(img)
        depth_bin = cv2.threshold(img, d_thr + RANGE_DELTA_M, 255, cv2.THRESH_BINARY_INV)
        return depth_bin[1]

    @classmethod
    def draw_contours_info(cls, cont, img_contours, merge_points):
        for i in range(len(cont[0])):
            c = cont[0][i]
            brect = cv2.boundingRect(c)
            org = (brect[0] + brect[2] / 2, brect[1] + brect[3] / 2)
            fnt = cv2.FONT_HERSHEY_PLAIN
            clr = RGB(0, 0, 255)
            cv2.putText(img_contours, str(i), org, fnt, 1.0, clr)

        if merge_points != None:
            for p in merge_points:
                cv2.circle(img_contours, tuple(p), 10, clr)

    @classmethod
    def dump_contours(cls, cont, fname):
        d = {}
        for i in range(len(cont)):
            li = len(cont[i])
            for j in [ x for x in range(len(cont)) if x != i ]:
                lj = len(cont[j])
                d[str(i) + 'B ' + str(j) + 'B'] = dist(cont[i][0][0], cont[j][0][0])
                d[str(i) + 'B ' + str(j) + 'E'] = dist(cont[i][0][0], cont[j][lj - 1][0])
                d[str(i) + 'E ' + str(j) + 'B'] = dist(cont[i][li - 1][0], cont[j][0][0])
                d[str(i) + 'E ' + str(j) + 'E'] = dist(cont[i][li - 1][0], cont[j][lj - 1][0])

        pickle.dump(d, file(fname, 'w'))

    @classmethod
    def check_contour(cls, c, ci):
        for j in range(len(c) - 1):
            p1 = c[j]
            p2 = c[j + 1]
            d = dist(p1[0], p2[0])
            if d > DepthDetector.MIN_DIST:
                raise Exception('BOOM in i%d j%d d%d %d %d %d %d' % (j,
                 j + 1,
                 d,
                 p1[0][0],
                 p1[0][1],
                 p2[0][0],
                 p2[0][1]))

        for i in range(len(c) - 1):
            for j in range(len(c) - 1):
                if j == i:
                    continue
                if np.all(c[i] == c[j]):
                    raise Exception('Equal contour elements(%d): %d and %d' % (ci, i, j))

    @classmethod
    def check_contours(cls, cont):
        for i in range(len(cont)):
            c = cont[i]

    @classmethod
    def prune_contours(cls, cont):
        for i in range(len(cont)):
            c = cont[i]
            new_c = np.array(c[0], dtype='int32')
            for j in range(len(c) - 1):
                for k in range(j + 1, len(c)):
                    pass

    @classmethod
    def get_contours(cls, img_fore):
        img_ed_canny = cv2.Canny(img_fore, 10, 30)
        img_ed = max_filter(img_ed_canny)
        cont = cv2.findContours(img_ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pickle.dump(cont[0], file('c0', 'w'))
        cont = ([ c for c in cont[0] if len(c.shape) == 3 and len(c) > 600 ], cont[1])
        img_contours0 = np.zeros([480, 640, 3], dtype=np.uint8)
        cv2.drawContours(img_contours0, cont[0], -1, RGB(255, 255, 0))
        img_contours = np.zeros([480, 640, 3], dtype=np.uint8)
        cv2.drawContours(img_contours, cont[0], -1, RGB(255, 0, 0))
        return (cont[0], img_contours, img_contours0)

    @classmethod
    def find_close_conts(cls, cont):
        for i in range(len(cont)):
            c_i0 = cont[i][0]
            c_il = cont[i][len(cont[i]) - 1]
            for j in [ x for x in range(len(cont)) if x != i ]:
                c_j0 = cont[j][0]
                c_jl = cont[j][len(cont[j]) - 1]
                if dist(c_i0[0], c_j0[0]) < DepthDetector.MIN_DIST:
                    return (i,
                     j,
                     0,
                     0)
                if dist(c_i0[0], c_jl[0]) < DepthDetector.MIN_DIST:
                    return (i,
                     j,
                     0,
                     len(cont[j]) - 1)
                if dist(c_il[0], c_jl[0]) < DepthDetector.MIN_DIST:
                    return (i,
                     j,
                     len(cont[i]) - 1,
                     len(cont[j]) - 1)
                if dist(c_il[0], c_j0[0]) < DepthDetector.MIN_DIST:
                    return (i,
                     j,
                     len(cont[i]) - 1,
                     0)

        return (-1, -1, -1, -1)

    @classmethod
    def merge_two_contours(cls, cont, i, j, i_pos, j_pos):
        a = cont[i] if i_pos != 0 else cont[i][::-1]
        b = cont[j] if j_pos == 0 else cont[j][::-1]
        middle, _ = build_line(cont[i][i_pos][0], cont[j][j_pos][0])
        ret = np.append(a, middle, axis=0)
        ret = np.append(ret, b, axis=0)
        return ret

    @classmethod
    def merge_contours(cls, cont, merge_points):
        try:
            cls.merge_i += 1
        except:
            cls.merge_i = 0

        cls.dump_contours(cont, 'd%03d' % cls.merge_i)
        i, j, i_pos, j_pos = cls.find_close_conts(cont)
        if i == -1:
            return (cont, merge_points)
        new_cont = [ cont[k] for k in range(len(cont)) if k != i and k != j ]
        print 'Merging %d %d' % (i, j)
        merged_pair = cls.merge_two_contours(cont, i, j, i_pos, j_pos)
        merge_points = np.append(merge_points, [cont[i][i_pos][0], cont[j][j_pos][0]], axis=0)
        new_cont.append(merged_pair)
        return cls.merge_contours(new_cont, merge_points)

    def plot_dist_hist(self, imgs_float, fname):
        v = np.var(imgs_float, axis=0)
        v_min = np.min(v)
        v_max = np.max(v)
        cv2.imwrite('depth_var.png', (v * 255.0 / v_max).astype('uint8'))
        large_v = np.where(v > 0.5)
        v_step = (v_max - v_min) / 3
        cmap = mpl.colors.ListedColormap(['black', 'blue', 'red'])
        bounds = [v_min,
         v_min + v_step,
         v_min + v_step * 2,
         v_max]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.clf()
        img = plt.imshow(v, interpolation='nearest', cmap=cmap, norm=norm)
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[v_min, 0, v_max])
        plt.savefig(fname)

    def save_stat_pixel_data(self, imgs):
        f = file(self.dist_pixel_fname, 'w')
        u, v = self.stat_pixel_pos
        for i in range(len(imgs)):
            f.write(str(imgs[i][v][u]) + '\n')

        f.close()

    def write_dist_hist_img(self, img_float):
        if self.dist_hist_imgs_i == self.dist_hist_sample_size:
            self.save_stat_pixel_data(self.dist_hist_imgs)
            fname = self.dist_hist_fname_base + '_%02d' % self.dist_hist_sample_number
            self.dist_hist_sample_number += 1
            self.plot_dist_hist(self.dist_hist_imgs, fname)
            self.dist_hist_imgs = np.zeros([self.dist_hist_sample_size, 480, 640])
            self.dist_hist_imgs_i = 0
        else:
            self.dist_hist_imgs[self.dist_hist_imgs_i] = img_float
            self.dist_hist_imgs_i += 1

    def get_convex_hulls(self, contours):
        MIN_CONTOUR_LEN = 100
        curv = []
        for c in contours:
            if len(c) > MIN_CONTOUR_LEN:
                curv.append(cv2.convexHull(c))

        return curv

    def draw_convex_hulls(self, img, conv):
        for c in conv:
            prev_p = c[0]
            for p in c[1:]:
                cv2.line(img, tuple(prev_p[0]), tuple(p[0]), RGB(0, 255, 0))
                prev_p = p

    def draw_debug_info(self, img):
        CROSS_SIZE = 20
        c = self.stat_pixel_pos
        p1 = (c[0], c[1] - CROSS_SIZE / 2)
        p2 = (c[0], c[1] + CROSS_SIZE / 2)
        p3 = (c[0] - CROSS_SIZE / 2, c[1])
        p4 = (c[0] + CROSS_SIZE / 2, c[1])
        color = RGB(0, 0, 255)
        cv2.line(img, p1, p2, color)
        cv2.line(img, p3, p4, color)

    def filter_edge_depth_simple(self, img):
        img[np.where(np.isnan(img))] = self.MAX_RANGE

    def filter_edge_depth(self, img):
        self.new_img_filter_buf = img
        nans = np.where(np.isnan(img))
        self.new_img_filter_buf[nans] = self.img_filter_buf[nans]
        self.img_filter_buf = self.new_img_filter_buf
        return self.img_filter_buf

    def rgb_cb(self, msg):
        pass

    def depth_cb(self, msg):
        try:
            img = bridge.imgmsg_to_cv(msg, 'passthrough')
            img = np.asarray(img)
        except CvBridgeError as e:
            print >> stderr, 'Cannot convert from ROS msg to CV image:', e

        x = self.mouse_x
        y = self.mouse_y
        self.filter_edge_depth_simple(img)
        print 'Depth(%d, %d): %f' % (x, y, img[y][x])
        if self.write_dist_hist and self.depth_frame_cnt % 3 == 0:
            self.write_dist_hist_img(img)
        self.depth_frame_cnt += 1
        img_fore = self.get_foreground(img).astype('uint8')
        cv2.imwrite('img_fore.png', img_fore)
        contours, img_contours, img_contours0 = DepthDetector.get_contours(img_fore)
        hulls = self.get_convex_hulls(contours)
        self.draw_debug_info(img_contours)
        cv2.imshow('wnd_orig', (img * 100).astype('uint8'))
        cv2.imshow('wnd_prob', img_fore)
        cv2.imshow('wnd_contours', img_contours)
        cv2.imwrite('img_contours%03d.png' % self.depth_frame_cnt, img_contours)
        cv2.imwrite('img0contours%03d.png' % self.depth_frame_cnt, img_contours0)
        ch = cv2.waitKey(10)
        if ch == 27:
            rospy.signal_shutdown('Quit')
        elif ch == ord(' '):
            cv2.imwrite('img_prob.png', img_prob)

    def run(self):
        rospy.spin()

    def on_shutdown(self):
        cv2.destroyAllWindows()
        cv2.waitKey(100)


def depth_detector(topic_name, thr_mode):
    dpdet = DepthDetector(topic_name, thr_mode)
    dpdet.run()


def do_track_hand():
    assert len(argv) >= 2, 'USAGE: track_hand.py <threshold_mode>'
    if argv[1] == 'depth_auto':
        thr_mode = ThresholdMode.DEPTH_AUTO
    elif argv[1] == 'depth_manual':
        thr_mode = ThresholdMode.DEPTH_MANUAL
    else:
        raise Exception('Illegal threshold_mode %s, use DEPTH_AUTO' % argv[1])
    depth_detector(DEF_TOPIC_NAME, thr_mode)


def test():
    img_fore = cv2.imread('data/img_fore.png')
    img_fore = cv2.cvtColor(img_fore, cv2.COLOR_BGR2GRAY)
    contours, img, img_old = DepthDetector.get_contours(img_fore)
    cv2.imwrite('d.png', img)
    cv2.imwrite('d0.png', img_old)


def test2():
    img = np.zeros([320, 240], dtype='uint8')
    l, img = build_line([1, 30], [20, 1], True)
    cv2.imwrite('d.png', img)


def test3():
    img = cv2.imread('d0.png')
    cv2.namedWindow('c0')
    c0 = pickle.load(file('c0'))
    i = 0
    for ci in range(len(c0)):
        c = c0[ci]
        for pi in range(len(c)):
            p = c[pi][0]
            img_i = img.copy()
            s = 'Contour %02d Point %03d of %03d' % (ci, pi, len(c))
            cv2.putText(img_i, s, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, RGB(0, 255, 0))
            cv2.circle(img_i, tuple(p), 10, RGB(0, 255, 255))
            cv2.imshow('c0', img_i)
            cv2.waitKey(300 - int(1200.0 / 300 * len(c)))

    while cv2.waitKey(100) != 27:
        pass

    cv2.destroyWindow('c0')
    cv2.waitKey(100)
    cv2.waitKey(100)


def test4():
    img = cv2.imread('canny.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ret = max_filter(img)
    cv2.imwrite('filt.png', img_ret)


if __name__ == '__main__':
    #do_track_hand()
    test()
