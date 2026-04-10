/**
 * webcam_app.cpp — implementations for webcam_app.h
 */

#include "webcam_app.h"
#include "tracker.h"

#include <chrono>
#include <ctime>
#include <iostream>

CvCamera::CvCamera(int index)
{
    // Try the fastest backend on Linux first; fall back to auto-selection.
    cap_.open(index, cv::CAP_V4L2);
    if (!cap_.isOpened())
        cap_.open(index);

    if (cap_.isOpened()) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT,  720);
        std::cout << "Camera " << index << " opened at "
                  << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
    }
}

bool CvCamera::isOpened() const { return cap_.isOpened(); }

bool CvCamera::read(cv::Mat& frame) { return cap_.read(frame); }

void CvCamera::release() { cap_.release(); }


void on_mouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN) return;

    auto* state          = static_cast<AppState*>(userdata);
    state->click_pos     = {x, y};
    state->save_requested = true;
}


std::string timestamp_str()
{
    using namespace std::chrono;

    auto now = system_clock::now();
    auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::time_t t = system_clock::to_time_t(now);
    std::tm     tm{};
    localtime_r(&t, &tm);

    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

    // Zero-pad milliseconds to exactly 3 digits (e.g. 7 → "007").
    char ms_buf[4];
    std::snprintf(ms_buf, sizeof(ms_buf), "%03d", static_cast<int>(ms.count()));
    return std::string(buf) + "_" + ms_buf;
}


std::string draw_and_save(const cv::Mat& frame, cv::Point pos,
                          const std::string& save_dir)
{
    fs::create_directories(save_dir);

    cv::Mat annotated = frame.clone();
    cv::circle(annotated, pos, CIRCLE_RADIUS, CIRCLE_COLOR,
               CIRCLE_THICKNESS, cv::LINE_AA);

    std::string path = save_dir + "/" + timestamp_str() + ".png";
    if (!cv::imwrite(path, annotated)) {
        std::cerr << "Error: failed to write " << path << "\n";
        return {};
    }
    std::cout << "Saved: " << path << "  [click at " << pos << "]\n";
    return path;
}


int run(ICamera& cam, ITracker& tracker, DisplaySink& display,
        const std::string& save_dir, int max_empty)
{
    if (!cam.isOpened()) {
        std::cerr << "Error: camera is not open\n";
        return 1;
    }

    AppState state{};
    display.setup(WINDOW_NAME, &state);

    int empty_streak = 0;

    while (true) {
        // grab frame
        if (!cam.read(state.frame) || state.frame.empty()) {
            if (++empty_streak >= max_empty) {
                std::cerr << "Error: too many consecutive empty frames, giving up\n";
                break;
            }
            continue;
        }
        empty_streak = 0;

        // handle pending click: (re)init tracker and save
        if (state.save_requested) {
            state.save_requested = false;
            tracker.init(state.frame, state.click_pos);
            draw_and_save(state.frame, state.click_pos, save_dir);

            cv::Mat annotated = state.frame.clone();
            cv::circle(annotated, state.click_pos, CIRCLE_RADIUS,
                       CIRCLE_COLOR, CIRCLE_THICKNESS, cv::LINE_AA);
            display.show_annotated(WINDOW_NAME, annotated, 300);
            continue;
        }

        // advance tracker and overlay circle on the live frame
        cv::Mat display_frame = state.frame.clone();
        if (tracker.is_active()) {
            cv::Point tracked_pos;
            if (tracker.update(state.frame, tracked_pos)) {
                cv::circle(display_frame, tracked_pos, CIRCLE_RADIUS,
                           CIRCLE_COLOR, CIRCLE_THICKNESS, cv::LINE_AA);
            } else {
                tracker.reset();  // lost; wait for next click
            }
        }

        // show frame and check for quit keys
        int key = display.show_frame(WINDOW_NAME, display_frame);
        if (key == 'q' || key == 'Q' || key == 27 /* Esc */) break;
    }

    cam.release();
    return 0;
}
