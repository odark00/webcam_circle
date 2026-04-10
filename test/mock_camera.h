#pragma once

/**
 * mock_camera.h — gmock stand-in for ICamera.
 *
 * Include this only in test binaries, never in production code.
 */

#include "webcam_app.h"

#include <gmock/gmock.h>

class MockCamera : public ICamera {
public:
    MOCK_METHOD(bool, isOpened, (), (const, override));
    MOCK_METHOD(bool, read, (cv::Mat& frame), (override));
    MOCK_METHOD(void, release, (), (override));
};
