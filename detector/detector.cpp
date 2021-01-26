#include "detector.h" 
#include <iostream>

using namespace cv;
using namespace std;
using namespace dlib;

CascadeDetector::CascadeDetector()
{
    detector.load("data/haarcascade_frontalface_alt.xml"); 
}

std::vector<Rect> CascadeDetector::detectFaces(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray); //not really sure what this does
    detector.detectMultiScale(frame_gray, faces);
    return faces;
}

void CascadeDetector::drawBoxAroundFaces(Mat frame, std::vector<Rect> faces)
{
    Detector::drawBoxAroundFaces(frame, faces, "Cascade Classifier", Scalar(0,255,0), Scalar(0,255,0));
}

HOGDetector::HOGDetector()
{
    detector = get_frontal_face_detector(); 
}

std::vector<Rect> HOGDetector::detectFaces(Mat frame)
{
    std::vector<Rect> faces;
    cv_image<bgr_pixel> cimg(frame); // Convert from opencv to dlib
    std::vector<dlib::rectangle> dlib_faces = detector(cimg);
    for_each(begin(dlib_faces), end(dlib_faces), [&faces](dlib::rectangle dlib_face) mutable
            {
            faces.push_back(
                    Rect(
                        Point(dlib_face.left(), dlib_face.top()),
                        Point(dlib_face.right(), dlib_face.bottom())
                        )
                    );
            });

    return faces;
}

void HOGDetector::drawBoxAroundFaces(Mat frame, std::vector<cv::Rect> faces)
{
    Detector::drawBoxAroundFaces(frame, faces, "Dlib HOG", Scalar(0,0,255), Scalar(0,0,255));
}
