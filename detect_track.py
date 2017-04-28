#!/usr/bin/env python

import numpy as np
#/usr/bin/env python
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
            cv2.putText(frame, '%d %s %0.0f' % (index, label, confidence * 100) + '%', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('visualize', frame)
        key = cv2.waitKey(1000)
        #if key == 10:
        #    print('No target we need.')
        #    cv2.destroyAllWindows()
        #    return 1 
        #else:
        #    cv2.destroyAllWindows()
        return self.choose_target()
 
    def choose_target(self):
        self.target = {}
        self.target['id'] = int(input("Enter Object ID: "))
        print('choose target id %d\n' % self.target['id'])
        result = self.detect_results[-1][self.target['id']]
        top_left = (result['topleft']['x'], result['topleft']['y']) 
        bottom_right = (result['bottomright']['x'], result['bottomright']['y']) 
        self.target['coordinate'] = top_left + bottom_right
        return 0
 
    def get_hist_feature(self, frame, coordinate):
        print(frame.shape)
        print(coordinate)
        (x0, y0, x1, y1) = coordinate
        # set up the ROI for tracking
        roi = frame[x0 : x1, y0 : y1]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        return roi_hist
        
    def track(self, frame, track_window, roi_hist):
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, update_track_window = cv2.CamShift(dst, track_window, term_crit)
        print(ret, update_track_window)
        print(ret)
        print(update_track_window)
        return (ret, update_track_window)

if __name__ == '__main__':
    tp = TrackPipeline()
    cap = cv2.VideoCapture(0)
    roi_hist = None
    while True:
        ret, frame = cap.read()
        results = tp.detect(frame)
        ret = tp.visulize(copy.deepcopy(frame), results)
        if ret != 0:
            continue
        else:
            roi_hist = tp.get_hist_feature(frame, tp.target['coordinate'])
            break
    while True:
        print('Begin tracking')
        ret, frame = cap.read()
        (ret, tp.target['coordinate']) = tp.track(frame, tp.target['coordinate'], roi_hist)
        #roi_hist = tp.get_hist_feature(frame, tp.target['coordinate'])
        cv2.rectangle(frame, tp.target['coordinate'][0:2], tp.target['coordinate'][2:4], (0, 255, 0), 3)
        cv2.imshow('tracking', frame)
        cv2.waitKey(2000)
