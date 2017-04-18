#!/usr/bin/env python

import numpy as np
import cv2
from darkflow.net.build import TFNet
import json
import copy

class TrackPipeline(object):

    def __init__(self):
        options = {"model": "cfg/yolo.cfg", "load": "weights/yolo.weights", "threshold": 0.1}
        self.tfnet = TFNet(options)
        self.detect_results = []

    def detect(self, frame): 
        self.detect_results.append(self.tfnet.return_predict(frame))
        return self.detect_results[-1]        

    def visulize(self, frame, results):
        for index, result in enumerate(results):
            # format as {'label': 'tvmonitor', 'confidence': 0.37281746, 'topleft': {'x': 313, 'y': 384}, 'bottomright': {'x': 510, 'y': 479}}
            label = result['label']
            confidence = result['confidence']
            top_left = (result['topleft']['x'], result['topleft']['y']) 
            bottom_right = (result['bottomright']['x'], result['bottomright']['y']) 
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
            cv2.putText(frame, '%d %s %0.0f' % (index, label, confidence * 100) + '%', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
        cv2.imshow('visualize', frame)
        key = cv2.waitKey(1000)
        if key == 10:
            print('No target we need.')
            cv2.destroyAllWindows()
            return 1 
        else:
            #return self.choose_target()
            return self.choose_label()

    def choose_label(self):
        labels = [result['label'] for result in self.detect_results[-1]]
        try:
            self.target = {}
            labelname = input("Enter label name: ")
            self.target['id'] = labels.index(labelname)
            if self.target['id'] < 0:
                return 1
            result = self.detect_results[-1][self.target['id']]
            top_left = (result['topleft']['x'], result['topleft']['y']) 
            bottom_right = (result['bottomright']['x'], result['bottomright']['y']) 
            self.target['coordinate'] = top_left + bottom_right
            cv2.destroyAllWindows()
            return 0
        except:
            cv2.destroyAllWindows()
            return 1
        

    def choose_target(self):
        try:
            self.target = {}
            self.target['id'] = int(input("Enter Object ID: "))
            if self.target['id'] < 0:
                return 1
            result = self.detect_results[-1][self.target['id']]
            top_left = (result['topleft']['x'], result['topleft']['y']) 
            bottom_right = (result['bottomright']['x'], result['bottomright']['y']) 
            self.target['coordinate'] = top_left + bottom_right
            cv2.destroyAllWindows()
            return 0
        except:
            cv2.destroyAllWindows()
            return 1
 
    def get_hist_feature(self, frame, coordinate):
        (x0, y0, x1, y1) = coordinate
        # set up the ROI for tracking
        roi = frame[y0 : y1, x0 : x1]
        cv2.imshow('object', roi)
        cv2.waitKey(2000)
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
        #mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255., 255., 255.)))
        #roi_hist = cv2.calcHist([hsv_roi], [0], mask, [32], [0,180])
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [16], [0,180])
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        return roi_hist
        
    def camshift_track(self, frame, track_window, roi_hist):
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1 )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, update_track_window = cv2.CamShift(dst, track_window, term_crit)
        return (ret, update_track_window)


    def box2xywh(self, window):
        #x, y, w, h
        return (window[0], window[1], window[2] - window[0], window[3] - window[1])
				

    def init_tracker(self, tracker_name, frame, box):
        self.tracker = cv2.Tracker_create(tracker_name)
        self.tracker.init(frame, box)
        
    def update_tracker(self, frame):
        return self.tracker.update(frame)
         


if __name__ == '__main__':
    tracker = 'KCF'
    tp = TrackPipeline()
    cap = cv2.VideoCapture(0)
    while True:
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        results = tp.detect(frame)
        ret = tp.visulize(copy.deepcopy(frame), results)
        if ret != 0:
            continue
        else:
            break
    if tracker == 'camshift':
        roi_hist = None
        roi_hist = tp.get_hist_feature(frame, tp.target['coordinate'])
        while True:
            ret, frame = cap.read()
            tracking_window = tp.box2xywh(tp.target['coordinate'])
            (ret, tracking_window) = tp.camshift_track(frame, tracking_window, roi_hist)
            #roi_hist = tp.get_hist_feature(frame, tp.target['coordinate'])
            cv2.rectangle(frame, (tracking_window[0], tracking_window[1]), (tracking_window[0] + tracking_window[2], tracking_window[1] + tracking_window[3]), (0, 255, 0), 3)
            cv2.imshow('tracking', frame)
            cv2.waitKey(10)
    elif tracker == 'KCF':
        box = tp.box2xywh(tp.target['coordinate'])
        tp.init_tracker(tracker, frame, box)
        while True:
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            (ret, tracking_window) = tp.update_tracker(frame)
            tracking_window = tuple([int(element) for element in tracking_window])
            cv2.rectangle(frame, (tracking_window[0], tracking_window[1]), (tracking_window[0] + tracking_window[2], tracking_window[1] + tracking_window[3]), (0, 255, 0), 3)
            cv2.imshow('tracking', frame)
            cv2.waitKey(1)
