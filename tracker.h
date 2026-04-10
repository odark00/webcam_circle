#pragma once

/**
 * tracker.h — ITracker interface and concrete implementations.
 *
 * Swap tracking strategies by passing a different ITracker to run().
 *
 * Current implementations:
 *   LKTracker  — Lucas-Kanade sparse optical flow (fast, good for single points)
 *
 * To add a new tracker (e.g. CSRT bounding-box):
 *   1. Subclass ITracker
 *   2. Implement init(), update(), is_active(), reset()
 *   3. Construct it in main.cpp and pass it to run()
 */

#include <opencv2/opencv.hpp>
#include <vector>

class ITracker {
public:
    virtual ~ITracker() = default;

    /**
     * Start (or restart) tracking at @p point in @p frame.
     * Must be called before the first update().
     */
    virtual void init(const cv::Mat& frame, cv::Point point) = 0;

    /**
     * Advance tracking to the next @p frame.
     * @param out_point  Receives the new estimated position on success.
     * @return true if the point was successfully tracked; false if lost.
     *         On false the caller should call reset() and wait for a new click.
     */
    virtual bool update(const cv::Mat& frame, cv::Point& out_point) = 0;

    /** True between init() and a failed update() (or reset()). */
    virtual bool is_active() const = 0;

    /** Stop tracking; the next init() will start fresh. */
    virtual void reset() = 0;
};

// LKTracker — Lucas-Kanade sparse optical flow

/**
 * Tracks a single clicked point using cv::calcOpticalFlowPyrLK.
 *
 * Strengths : very fast, works well for textured regions.
 * Weaknesses: can drift over time; loses the point if it leaves the frame
 *             or is occluded.
 */
class LKTracker : public ITracker {
public:
    void init(const cv::Mat& frame, cv::Point point) override;
    bool update(const cv::Mat& frame, cv::Point& out_point) override;
    bool is_active() const override { return active_; }
    void reset() override;

private:
    cv::Mat                  prev_gray_;   // grayscale of the previous frame
    std::vector<cv::Point2f> points_;      // tracked points in prev frame
    bool                     active_{false};
};
