#include "main.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>

using namespace std;
using namespace cv;


CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

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
        waitKey(100); //wait for 30 ms
    }
    return 0;
}

void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray); //not really sure what this does
    
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for_each(begin(faces), end(faces), [frame](Rect face)
            {
                rectangle(frame, Rect(Point(face.x,face.y), Point(face.x+face.width,face.y+face.height)), Scalar(0, 255, 0), 4);
            });

    imshow( "Capture - Face detection", frame );
}
