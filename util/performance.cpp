#include "performance.h"

CountsPerSec::CountsPerSec() : start(std::chrono::steady_clock::now()), num_occurrences(0) {}

void CountsPerSec::increment()
{
    num_occurrences++;
}

float CountsPerSec::counts_per_sec()
{
    std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
    std::chrono::duration<float> difference = end - start;

    return num_occurrences / difference.count() ;
}

void put_iterations_per_sec(cv::Mat frame, float iterations_per_sec)
{
    // std::string text = std::format("{} iterations/sec", iterations_per_sec); compilers don't support this yet
    std::string text = fmt::format("{:.2f} iterations/sec", iterations_per_sec);
    cv::putText(frame, text, cv::Point(50, 600), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(128,128,128));
}
