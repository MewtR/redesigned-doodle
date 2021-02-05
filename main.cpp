#include "main.h"
#include "dlib/opencv/cv_image.h"

using namespace std;
using namespace cv;
using namespace dlib;

// From opencv
CascadeDetector cascade_detector; // calls my default constructor

// From dlib
HOGDetector hog_detector; // calls my default constructor

void detectAndDisplay(Mat frame);

int main()
{
    Mat snapshot;
    Mat snapshot_rgb;
    
    VideoCapture camera(0);

    std::vector<dlib::rectangle> faces; 
    if (! camera.isOpened())
    {
        cout << "Error opening video device " << endl;
        return -1;
    }
    
    
    while(camera.read(snapshot))
    {
        //detectAndDisplay(snapshot);
        // Convert to RGB
        cvtColor(snapshot, snapshot_rgb, COLOR_BGR2RGB);
        // convert to dlib style image
        cv_image<rgb_pixel> img(snapshot_rgb); 
        faces = detectFaces(img);
        cout << "Number of faces detected: "<< faces.size() << endl;
        waitKey(25);
    }
    
    return 0;
}

void detectAndDisplay(Mat frame)
{
    std::vector<Rect> cascade_faces = cascade_detector.detectFaces(frame);
    std::vector<Rect> hog_faces = hog_detector.detectFaces(frame);
    hog_detector.drawBoxAroundFaces(frame, hog_faces);
    cascade_detector.drawBoxAroundFaces(frame, cascade_faces);
    imshow( "Capture - Face detection", frame );
}
