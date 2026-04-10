/**
 * main.cpp — entry point for webcam_circle.
 *
 * Gets webcam stream and displays window and allows click selection for points tracking.
 *
 * Usage:
 *   ./webcam_circle [camera_index]   (default 0)
 */

#include "webcam_app.h"
#include "tracker.h"

#include <iostream>

int main(int argc, char* argv[])
{
    int index = (argc > 1) ? std::stoi(argv[1]) : 0;

    CvCamera cam(index);
    if (!cam.isOpened()) {
        std::cerr << "Error: cannot open camera " << index << "\n";
        return 1;
    }

    // Swap LKTracker for any other ITracker subclass here to change the algorithm.
    LKTracker tracker;

    // OpenCV GUI calls
    DisplaySink display{
        // setup: create window and register mouse callback
        [](const std::string& window, AppState* state) {
            cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback(window, on_mouse, state);
            std::cout << "Left-click on the video to mark a point and save.\n";
        },
        // show_frame: display live feed, return key code (or -1)
        [](const std::string& window, const cv::Mat& frame) -> int {
            cv::imshow(window, frame);
            return cv::waitKey(30);
        },
        // show_annotated: display the circle-annotated frame briefly
        [](const std::string& window, const cv::Mat& frame, int delay_ms) {
            cv::imshow(window, frame);
            cv::waitKey(delay_ms);
        }
    };

    int result = run(cam, tracker, display);
    cv::destroyAllWindows();
    return result;
}
