#include "recognizor.h"

using namespace cv;
using namespace dlib;

dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

std::vector<dlib::rectangle> detectFaces(Mat frame)
{
    cv_image<bgr_pixel> cimg(frame);
    std::vector<dlib::rectangle> faces = detector(cimg);
    return faces;
}
