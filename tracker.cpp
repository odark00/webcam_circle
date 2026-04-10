/**
 * tracker.cpp — ITracker implementations.
 */

#include "tracker.h"
#include <iostream>

void LKTracker::init(const cv::Mat& frame, cv::Point point)
{
    cv::cvtColor(frame, prev_gray_, cv::COLOR_BGR2GRAY);
    points_ = { static_cast<cv::Point2f>(point) };
    active_ = true;
}

bool LKTracker::update(const cv::Mat& frame, cv::Point& out_point)
{
    if (!active_ || points_.empty()) return false;

    cv::Mat curr_gray;
    cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> next_points;
    std::vector<uchar>       status;
    std::vector<float>       err;

    // LK parameters: 3-level pyramid, 21×21 window, 30 iterations / 0.01 eps
    cv::calcOpticalFlowPyrLK(
        prev_gray_, curr_gray,
        points_,    next_points,
        status,     err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    if (!status.empty() && status[0]) {
        out_point = static_cast<cv::Point>(next_points[0]);
        points_    = next_points;   // advance: current becomes previous
        prev_gray_ = curr_gray;
        return true;
    }

    // Tracking lost
    std::cout << "Tracker: point lost\n";
    active_ = false;
    return false;
}

void LKTracker::reset()
{
    active_ = false;
    points_.clear();
    prev_gray_ = cv::Mat{};
}
