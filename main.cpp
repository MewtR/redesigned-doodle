#include "main.h"


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
    Mat pic = imread("data/randos.png");
    Mat snapshot;
    
    VideoCapture camera(0);

    if (! camera.isOpened())
    {
        cout << "Error opening video device " << endl;
        return -1;
    }
    camera.read(snapshot);
    

    
    /*
    while(camera.read(snapshot))
    {
    
        detectAndDisplay(snapshot);
        
        waitKey(50);
        break;
    }
    */
    detectAndDisplay(pic);
    
    cout << "Pic type is: "<< pic.type() << endl;
    cout << "Pic depth is: "<< pic.depth() << endl;
    cout << "Snapshot type is: "<< snapshot.type() << endl;
    cout << "Snapshot depth is: "<< snapshot.depth() << endl;
    return 0;
}

void detectAndDisplay(Mat frame)
{
    //std::vector<Rect> cascade_faces = cascade_detector.detectFaces(frame);
    std::vector<Rect> hog_faces = hog_detector.detectFaces(frame);
    hog_detector.drawBoxAroundFaces(frame, hog_faces);
    //cascade_detector.drawBoxAroundFaces(frame, cascade_faces);
    imshow( "Capture - Face detection", frame );
    //waitKey(0);
    imwrite("pic.png", frame);
}
