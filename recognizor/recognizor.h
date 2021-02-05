#ifndef RECOGNIZOR_H
#define RECOGNIZOR_H
#include "opencv2/core.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>

std::vector<dlib::rectangle> detectFaces(cv::Mat);
std::vector<dlib::matrix<dlib::rgb_pixel>> normalize(std::vector<dlib::rectangle>);
void setup(void);
#endif
