# import os, shutil
# from xml.etree.ElementTree import parse
# import tensorflow as tf
# import cv2
# import numpy as np
# def parse_function(filename, label) :
#     image_string = tf.io.read_file(filename)
# filename = './casia/img/CASIA_0000099_018.jpg'
# img = tf.io.read_file(filename)
# img = tf.image.decode_jpeg(img)
# img = tf.image.convert_image_dtype(img, tf.float32)
# img = np.array(img*255)
# # print(img)
# cv2.imshow('abc', img)

a = 5
c = a % 3
print(c)