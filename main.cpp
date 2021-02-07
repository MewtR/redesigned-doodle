#include "main.h"
#include "dlib/dir_nav/dir_nav_extensions.h"

using namespace std;
using namespace cv;
using namespace dlib;

// From opencv
CascadeDetector cascade_detector; // calls my default constructor

// From dlib
HOGDetector hog_detector; // calls my default constructor

void detectAndDisplay(Mat);
void train(Mat, string, std::map<string, matrix<float,0,1>>);

int main()
{
    Mat snapshot;
    Mat snapshot_rgb;
    Mat pic = imread("data/pictures/lemine.png");
    Mat pic_rgb;

    std::map<string, matrix<float,0,1>> known_faces;

        // dlib also has a function to check if a file exists.
        if (filesystem::exists("data/knownfaces/faces.dat"))
        {
            deserialize("data/knownfaces/faces.dat") >> known_faces;
            for (auto const& x: known_faces)
            {
                cout << x.first << endl;
            }
        }
    
    VideoCapture camera(0);

    std::vector<dlib::rectangle> faces; 
    std::vector<matrix<rgb_pixel>> normalized_faces; 
    std::vector<matrix<float,0,1>> face_descriptors; 
    if (! camera.isOpened())
    {
        cout << "Error opening video device " << endl;
        return -1;
    }
    // Load pre trained models
    setup();
    //train(pic, "Aghiles", known_faces);
    
    while(camera.read(snapshot))
    {
        //detectAndDisplay(snapshot);
        // Convert to RGB
        cvtColor(pic, pic_rgb, COLOR_BGR2RGB);
        // convert to dlib style image
        cv_image<rgb_pixel> img(pic_rgb); 
        faces = detectFaces(img);
        normalized_faces = normalize(faces, img);
        face_descriptors = convertToVector(normalized_faces);
        cout << "Face descriptor size: "<< face_descriptors.size() << endl;
        for (auto const& known_face : known_faces)
        {
            cout << "Distance between detected face and " << known_face.first << ": "<< length(known_face.second - face_descriptors[0])  << endl;
        }
        imshow( "Capture - Face detection", pic_rgb );
        waitKey(0);
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
void train(Mat frame, String label, std::map<string, matrix<float,0,1>> known_faces)
{
        Mat frame_rgb;

        std::vector<dlib::rectangle> faces; 
        std::vector<matrix<rgb_pixel>> normalized_faces; 
        std::vector<matrix<float,0,1>> face_descriptors; 

        cvtColor(frame, frame_rgb, COLOR_BGR2RGB);
        cv_image<rgb_pixel> img(frame_rgb); 
        faces = detectFaces(img);
        if (faces.size() != 1)
        {
            cout << "Train with only one face" << endl;
            return;
        }
        normalized_faces = normalize(faces, img);
        face_descriptors = convertToVector(normalized_faces); //has a size of 1 because the input should only have one face
        cout << "Face descriptor size: "<< face_descriptors.size() << endl;
        std::map<string, matrix<float,0,1>>::iterator element = known_faces.find(label);
        if (element != known_faces.end())
        {
            // element already exists, delete it and save new one
            known_faces.erase(element);
        }
        known_faces.insert({label, face_descriptors[0]});
        cout << "Saving..." << endl;
        serialize("data/knownfaces/faces.dat") << known_faces;
}
