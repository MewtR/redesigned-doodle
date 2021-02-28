#ifndef PERFORMANCE_H
#define PERFORMANCE_H
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
// #include <format> compilers don't support this yet
#include <fmt/format.h> // Use lib instead

// taken from https://github.com/nrsyed/computer-vision/blob/master/multithread/CountsPerSec.py
class CountsPerSec{
    private: 
        int num_occurrences;
        std::chrono::time_point<std::chrono::steady_clock> start;
    public:
        CountsPerSec();
        void increment();
        float counts_per_sec();

};

void put_iterations_per_sec(cv::Mat, float iterations_per_sec);

#endif
