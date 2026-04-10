#pragma once

/**
 * webcam_app.h — declarations for the webcam circle-marking application.
 *
 * Separates the testable business logic from main() and from OpenCV GUI calls.
 *
 * Key abstractions:
 *   ICamera     — pure interface; CvCamera wraps the real webcam; tests use MockCamera
 *   ITracker    — pure interface; LKTracker is the default; swap via run() argument
 *   DisplaySink — std::function members for every GUI call so tests can stub them out
 *   AppState    — shared state between the mouse callback and the main loop
 */

#include <opencv2/opencv.hpp>
#include "tracker.h"

#include <filesystem>
#include <functional>
#include <string>

namespace fs = std::filesystem;


inline constexpr int    CIRCLE_RADIUS    = 20;
inline constexpr int    CIRCLE_THICKNESS = 3;
inline const     cv::Scalar CIRCLE_COLOR = {0, 255, 0};  // BGR → green
inline constexpr char   SAVE_DIR[]       = "captures";
inline constexpr char   WINDOW_NAME[]    =
    "Webcam  |  click to track & save  |  Q/Esc = quit";


/**
 * Shared state written by the mouse callback and read by the main loop.
 * Kept as a plain struct (no mutex) because OpenCV's HighGUI callback runs on
 * the same thread as waitKey().
 */
struct AppState {
    cv::Mat   frame;           // latest camera frame
    bool      save_requested{false};
    cv::Point click_pos{0, 0};
};

/**
 * Abstract camera interface.  The production code uses CvCamera (wraps
 * cv::VideoCapture).  Unit tests use MockCamera (gmock).
 */
class ICamera {
public:
    virtual ~ICamera() = default;
    virtual bool isOpened() const          = 0;
    virtual bool read(cv::Mat& frame)      = 0;  // true on success, false on error
    virtual void release()                 = 0;
};

// webcam wrapper

/**
 * Wraps cv::VideoCapture.  Constructor tries the V4L2 backend first (fastest
 * on Linux), then falls back to OpenCV's auto-selection.
 */
class CvCamera : public ICamera {
public:
    /** @param index  /dev/video<index> (default 0) */
    explicit CvCamera(int index = 0);

    bool isOpened() const override;
    bool read(cv::Mat& frame) override;
    void release() override;

private:
    cv::VideoCapture cap_;
};


/**
 * Holds every OpenCV GUI call the main loop makes, as std::function members.
 *
 * Production main() fills these with real imshow/waitKey lambdas.
 * Tests fill them with no-op lambdas (no display server required).
 *
 * setup()         — called once before the loop; creates window + mouse callback
 * show_frame()    — called each iteration; returns the waitKey() result (-1 = no key)
 * show_annotated()— called after a save; shows the circle briefly then returns
 */
struct DisplaySink {
    std::function<void(const std::string& window, AppState* state)>            setup;
    std::function<int (const std::string& window, const cv::Mat& frame)>       show_frame;
    std::function<void(const std::string& window, const cv::Mat& frame, int)>  show_annotated;
};


/** OpenCV mouse callback — records left-click position in AppState. */
void on_mouse(int event, int x, int y, int flags, void* userdata);

/** Returns a filename-safe timestamp: YYYYMMDD_HHMMSS_mmm */
std::string timestamp_str();

/**
 * Clones @p frame, draws a circle at @p pos, writes a PNG to @p save_dir.
 * @return  The full path of the saved file (empty string on write failure).
 */
std::string draw_and_save(const cv::Mat& frame, cv::Point pos,
                          const std::string& save_dir);


/**
 * Runs the webcam capture loop until the user presses Q/Esc.
 *
 * Click behaviour:
 *   - Initialises @p tracker at the clicked point.
 *   - Saves an annotated PNG to @p save_dir.
 *   - Subsequent frames show a circle at the tracker's estimated position.
 *   - Click again to re-initialise tracking at a new point.
 *
 * @param cam        Camera source (real or mock).
 * @param tracker    Point tracker (real or mock); swap to change algorithm.
 * @param display    GUI callbacks (real or stubbed).
 * @param save_dir   Directory for saved PNG files.
 * @param max_empty  Max consecutive empty/failed frames before giving up.
 * @return 0 on clean exit, 1 on camera error.
 */
int run(ICamera& cam, ITracker& tracker, DisplaySink& display,
        const std::string& save_dir = SAVE_DIR,
        int max_empty = 100);
