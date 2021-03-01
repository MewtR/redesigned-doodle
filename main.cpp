#include "main.h"

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
    std::mutex snapshot_mutex;
    std::mutex faces_and_labels_mutex;

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
    std::map<dlib::rectangle, string> faces_and_labels; 

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

    std::thread process_snapshot([&grabbed, &snapshot, &snapshot_mutex, known_faces, &faces_and_labels_mutex, &faces_and_labels]() mutable
            {
                std::vector<dlib::rectangle> faces; 
                std::vector<matrix<rgb_pixel>> normalized_faces; 
                std::vector<matrix<float,0,1>> face_descriptors; 
                Mat snapshot_rgb;

                while(grabbed)
                {
                    { // separate scope so that lock is realease after
                        // Convert to RGB
                        cout << "Process thread convert to RGB" << endl;
                        std::lock_guard<std::mutex> lock(snapshot_mutex);
                        cout << "Process thread convert to RGB lock acquired" << endl;
                        cvtColor(snapshot, snapshot_rgb, COLOR_BGR2RGB);
                        // lock should be released here
                    }
                    cv_image<rgb_pixel> img(snapshot_rgb); 
                    faces = detectFaces(img);
                    cout << "NUMBER OF DETECTED FACES "<< faces.size() << endl;
                    if (faces.size() > 0)
                    {
                        normalized_faces = normalize(faces, img);
                        face_descriptors = convertToVector(normalized_faces);
                        //cout << "Face descriptor size: "<< face_descriptors.size() << endl;
                        //Using a regular for loop here because I'm banking on the fact that 
                        //if a face has index i, it's corresponding face_descriptor will also be at index i
                        //this seems to be a correct assumption
                        {
                            std::lock_guard<std::mutex> lock(faces_and_labels_mutex);
                            cout << "Process thread faces lock acquired" << endl;
                            if(!faces_and_labels.empty()) faces_and_labels.clear();
                            for (int i = 0; i < face_descriptors.size(); ++i) // one descriptor = one face
                            {
                                bool match = false;
                                for (auto const& known_face : known_faces)
                                {
                                    float distance = length(known_face.second-face_descriptors[i]);
                                    cout << "Distance between detected face and " << known_face.first << " is: " << distance << endl;
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
                        }
                        std::this_thread::yield();
                    }
                }
            });

    CountsPerSec cps;

    // Main thread
    while(grabbed)
    {
        {
            std::unique_lock<std::mutex> faces_lock(faces_and_labels_mutex);
            cout << "Main thread faces lock acquired" << endl;
            cout << "Number of faces to draw: " << faces_and_labels.size() << endl;
            std::lock_guard<std::mutex> lock(snapshot_mutex); // needed when waiting with condition variable
            cout << "Main thread put iterations lock acquired" << endl;
            if(!faces_and_labels.empty()){
                cout << "DRAWING FACES!!!!!!!!!!" << endl;;
                drawBoxAroundFaces(snapshot, faces_and_labels); 
                faces_lock.unlock();
            }
            //cv.wait(lock); // releases lock and waits until notified
            put_iterations_per_sec(snapshot, cps.counts_per_sec());
            imshow( "Capture - Face detection", snapshot);
        }
        cps.increment();
        waitKey(1);
    }

    capture_thread.join(); // never reached? because I sigint to terminate the program
    process_snapshot.join(); // never reached? because I sigint to terminate the program
    
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
