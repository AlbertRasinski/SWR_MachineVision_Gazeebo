#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import MachineVision as MV
import cv2

from cv_bridge import CvBridge

def callback(data):
    openCVBridge = CvBridge()
    
    currentFrame = openCVBridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2BGR)

    mv.uploadImage(currentFrame)
    detectionFrame = mv.runDetection()
    cv2.imshow("Result", detectionFrame)
    cv2.imshow("Camera", currentFrame)
    cv2.waitKey(10)


def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/rgb/image_raw", Image, callback,  queue_size = 1)
    rospy.spin()


if __name__ == '__main__':
    mv = MV.MachineVision(1000,100000,1,2,3)
    mv.addThresholdRange((0,100,0),(50,255,50))
    mv.addThresholdRange((100,100,0),(255,255,50))

    cv2.namedWindow("Result", 1)
    cv2.namedWindow("Camera", 2)

    listener()
