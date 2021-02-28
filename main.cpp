#include "main.h"
#include <thread>

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
    std::mutex snapshot_mutex;

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
    

    std::vector<dlib::rectangle> faces; 
    std::vector<matrix<rgb_pixel>> normalized_faces; 
    std::vector<matrix<float,0,1>> face_descriptors; 

    VideoCapture camera(0);
    if (! camera.isOpened())
    {
        cout << "Error opening video device " << endl;
        return -1;
    }
    // Load pre trained models
    setup();
    std::atomic<bool> grabbed = camera.read(snapshot);

    std::thread capture_thread([&camera, &snapshot, &grabbed, &snapshot_mutex]() mutable
            {
                while(grabbed)
                {
                    {
                    cout << "Capture thread " << endl;
                    std::lock_guard<std::mutex> lock(snapshot_mutex);
                    cout << "Capture thread lock acquired" << endl;
                    grabbed = camera.read(snapshot); // grabs new snapshot
                    }
                    std::this_thread::yield();
                }
            });

    CountsPerSec cps;
    while(grabbed)
    {
        
        std::map<dlib::rectangle, string> faces_and_labels; // this var should only have scope within the while loop
        //detectAndDisplay(snapshot);
        { // separate scope so that lock is realease after
            // Convert to RGB
            cout << "Main thread convert to RGB" << endl;
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            cout << "Main thread convert to RGB lock acquired" << endl;
            cvtColor(snapshot, snapshot_rgb, COLOR_BGR2RGB);
            // lock should be released here
        }
        // convert to dlib style image
        cv_image<rgb_pixel> img(snapshot_rgb); 
        faces = detectFaces(img);
        if (faces.size() > 0)
        {
            normalized_faces = normalize(faces, img);
            face_descriptors = convertToVector(normalized_faces);
            //cout << "Face descriptor size: "<< face_descriptors.size() << endl;
            //Using a regular for loop here because I'm banking on the fact that 
            //if a face has index i, it's corresponding face_descriptor will also be at index i
            //this seems to be a correct assumption
            for (int i = 0; i < face_descriptors.size(); ++i) // one descriptor = one face
            {
                bool match = false;
                for (auto const& known_face : known_faces)
                {
                    //cout << "Distance between detected face and " << known_face.first << ": "<< length(known_face.second - descriptor)  << endl;
                    float distance = length(known_face.second-face_descriptors[i]);
                    //cout << "Distance between detected face and " << known_face.first << " is: " << distance << endl;
                    if (distance < 0.5)
                    {
                        faces_and_labels.insert({faces[i], known_face.first});
                        match = true;
                        break; // found match leave
                    }
                }
                if (!match)
                {
                    faces_and_labels.insert({faces[i], "?????"});
                }
            }
            {
                cout << "Main thread draw boxes" << endl;
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                cout << "Main thread draw boxes lock acquired" << endl;
                drawBoxAroundFaces(snapshot, faces_and_labels);
            }
        }
        {
            cout << "Main thread put iterations" << endl;
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            cout << "Main thread put iterations lock acquired" << endl;
            put_iterations_per_sec(snapshot, cps.counts_per_sec());
        }
        {
            cout << "Main thread display snapshot" << endl;
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            cout << "Main thread display snapshot lock acquired" << endl;
            imshow( "Capture - Face detection", snapshot);
        }
        cps.increment();
        waitKey(1);
    }

    capture_thread.join(); // never reached? because I sigint to terminate the program
    
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
