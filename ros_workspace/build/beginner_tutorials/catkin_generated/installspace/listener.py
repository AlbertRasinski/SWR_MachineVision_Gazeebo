#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

import MachineVision as MV
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge



def callback(data):
        
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    mv.uploadImage(cv_image)
    img = mv.runDetection()
    cv2.imshow("Obrazek",img)
    cv2.waitKey(10)
    

    #system = SimilarityBase(base_image1, base_image2)
    #wyniki = system.calculateSimilarity(cv_image)

   
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/rgb/image_raw",
		Image, callback,  queue_size = 1)
	
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    mv = MV.MachineVision(1000,100000,1,2,3)
    mv.addThresholdRange((0,100,0),(50,255,50))
    mv.addThresholdRange((100,100,0),(255,255,50))
    cv2.namedWindow("Obrazek", 1)
    listener()