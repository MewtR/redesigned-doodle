#ifndef RECOGNIZOR_H
#define RECOGNIZOR_H
#include "dlib/opencv/cv_image.h"
#include "opencv2/core.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>

std::vector<dlib::rectangle> detectFaces(dlib::cv_image<dlib::rgb_pixel>);
std::vector<dlib::matrix<dlib::rgb_pixel>> normalize(std::vector<dlib::rectangle>, dlib::cv_image<dlib::rgb_pixel>);
void setup(void);
std::vector<dlib::matrix<float, 0, 1>> convertToVector(std::vector<dlib::matrix<dlib::rgb_pixel>>);
#endif
