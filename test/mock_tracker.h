#pragma once

/**
 * mock_tracker.h — gmock stand-in for ITracker.
 *
 * Include only in test binaries.
 */

#include "tracker.h"
#include <gmock/gmock.h>

class MockTracker : public ITracker {
public:
    MOCK_METHOD(void, init,   (const cv::Mat& frame, cv::Point point), (override));
    MOCK_METHOD(bool, update, (const cv::Mat& frame, cv::Point& out_point), (override));
    MOCK_METHOD(bool, is_active, (), (const, override));
    MOCK_METHOD(void, reset,  (), (override));
};
