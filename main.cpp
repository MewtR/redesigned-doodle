#include "main.h"

using namespace std;
using namespace cv;
using namespace dlib;

// From opencv
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

// From dlib
frontal_face_detector detector = get_frontal_face_detector();

void detectAndDisplay(Mat frame);

int main()
{
    Mat snapshot;
    Mat frame_gray;
    Mat frame_gray_eq;

    VideoCapture camera(0);

    if (! camera.isOpened())
    {
        cout << "Error opening video device " << endl;
        return -1;
    }
    //Load the cascades (pre trained models)
    if (!face_cascade.load("haarcascade_frontalface_alt.xml"))
    {
        cout << " Unable to load face cascade " << endl;
        return -1;
    }

    if (!eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml"))
    {
        cout << " Unable to load eyes cascade " << endl;
        return -1;
    }

    while(camera.read(snapshot))
    {
        detectAndDisplay(snapshot);
        waitKey(50);
    }
    return 0;
}

void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray); //not really sure what this does
    
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    // dlib image
    cv_image<bgr_pixel> cimg(frame); // Convert from opencv to dlib
    std::vector<dlib::rectangle> dlibFaces = detector(cimg);

    //Draw faces detected by dlib's face detector
    for_each(begin(dlibFaces), end(dlibFaces), [frame](dlib::rectangle face){

            cv::rectangle(frame, Point(face.left(), face.top()),Point(face.right(), face.bottom()), Scalar(0,0,255));
            putText(frame, "Dlib HOG", Point(face.left(), face.top()-3), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,0,255));
            });

    //Draw faces detected by opencv's cascade classifier
    for_each(begin(faces), end(faces), [frame](Rect face)
            {
            cv::rectangle(frame, face, Scalar(0, 255, 0), 1);
                putText(frame, "Cascade classifier" , Point(face.x,face.y-3) , FONT_HERSHEY_SIMPLEX, 1.0, Scalar (0,255,0));
            });

    imshow( "Capture - Face detection", frame );
}
