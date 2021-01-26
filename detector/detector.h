#ifndef DETECTOR_H
#define DETECTOR_H

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <vector>

class Detector{
    public:
        virtual std::vector<cv::Rect> detectFaces(cv::Mat) = 0;
        void drawBoxAroundFaces(cv::Mat frame, std::vector<cv::Rect> faces, std::string message, cv::Scalar box_color, cv::Scalar text_color)
        {
            for_each(begin(faces), end(faces), [frame, message, box_color, text_color](cv::Rect face)
            {
            cv::rectangle(frame, face, box_color, 1); //draw
            putText(frame, message , cv::Point(face.x,face.y-3) , cv::FONT_HERSHEY_SIMPLEX, 1.0, text_color); //write text
            });
        }
};

class CascadeDetector : public Detector{
    private:
        cv::CascadeClassifier detector;
    public:
        CascadeDetector();
        std::vector<cv::Rect> detectFaces(cv::Mat);
        void drawBoxAroundFaces(cv::Mat, std::vector<cv::Rect>);
};

class HOGDetector: public Detector{
    private:
        dlib::frontal_face_detector detector;
    public:
        HOGDetector();
        std::vector<cv::Rect> detectFaces(cv::Mat);
        void drawBoxAroundFaces(cv::Mat, std::vector<cv::Rect>);
};
#endif
