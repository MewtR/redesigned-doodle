#ifndef RECOGNIZOR_H
#define RECOGNIZOR_H
#include "opencv2/core.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

std::vector<dlib::rectangle> detectFaces(cv::Mat);
#endif
