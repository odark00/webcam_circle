/**
 * webcam_app_test.cpp — Google Test suite for webcam_app + tracker.
 *
 * Test groups:
 *   1. timestamp_str  — format and monotonicity
 *   2. on_mouse       — AppState mutation
 *   3. draw_and_save  — file I/O
 *   4. LKTracker      — init/update/reset behaviour
 *   5. run()          — main loop via MockCamera + MockTracker + no-op DisplaySink
 */

#include "mock_camera.h"   // MockCamera  (pulls in webcam_app.h + gmock)
#include "mock_tracker.h"  // MockTracker (pulls in tracker.h)

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <filesystem>
#include <regex>
#include <thread>
#include <chrono>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::NiceMock;
using ::testing::SetArgReferee;

namespace fs = std::filesystem;

// Returns a DisplaySink whose callbacks are all no-ops.
// show_frame() returns the values from `key_sequence` in order, then 'q'.
static DisplaySink make_noop_sink(std::vector<int> key_sequence = {})
{
    auto keys = std::make_shared<std::vector<int>>(std::move(key_sequence));
    auto idx  = std::make_shared<std::size_t>(0);

    return DisplaySink{
        [](const std::string&, AppState*) {},
        [keys, idx](const std::string&, const cv::Mat&) -> int {
            if (*idx < keys->size()) return (*keys)[(*idx)++];
            return 'q';
        },
        [](const std::string&, const cv::Mat&, int) {}
    };
}

// Returns a NiceMock<MockTracker> configured to be inactive (default state).
static void setup_inactive_tracker(NiceMock<MockTracker>& t)
{
    ON_CALL(t, is_active()).WillByDefault(Return(false));
}

// A 100×100 solid grey frame valid, non-empty, safe to pass to imwrite.
static cv::Mat make_grey_frame()
{
    return cv::Mat(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
}

// A frame with distinct local features, required by LK optical flow,
// which needs image gradients at the tracked point.
// cv::drawMarker draws a cross/star that has strong gradient at its centre,
// so clicking exactly at the marker position gives LK a reliable feature.
static cv::Mat make_textured_frame()
{
    cv::Mat f(200, 200, CV_8UC3, cv::Scalar(40, 40, 40));
    // Cross markers at the two positions used by the tracker tests.
    cv::drawMarker(f, {50, 50}, {255, 255, 255}, cv::MARKER_CROSS,      30, 2);
    cv::drawMarker(f, {80, 80}, {200, 100,  50}, cv::MARKER_TILTED_CROSS, 25, 2);
    // Extra background texture so optical flow has context.
    cv::rectangle(f, {10, 10}, {35, 35}, {80, 200, 80}, -1);
    return f;
}

// Common camera setup: open + always returns a grey frame.
static void setup_open_camera(NiceMock<MockCamera>& cam)
{
    ON_CALL(cam, isOpened()).WillByDefault(Return(true));
    ON_CALL(cam, read(_)).WillByDefault(DoAll(
        Invoke([](cv::Mat& f) { f = make_grey_frame(); }),
        Return(true)
    ));
    ON_CALL(cam, release()).WillByDefault(Return());
}


TEST(OnMouse, LeftClickSetsSaveRequestedAndClickPos)
{
    AppState state{};
    on_mouse(cv::EVENT_LBUTTONDOWN, 42, 77, 0, &state);

    EXPECT_TRUE(state.save_requested);
    EXPECT_EQ(state.click_pos, cv::Point(42, 77));
}

TEST(OnMouse, NonLeftClickDoesNotChangeState)
{
    AppState state{};
    on_mouse(cv::EVENT_MOUSEMOVE,   10, 10, 0, &state);
    on_mouse(cv::EVENT_RBUTTONDOWN, 10, 10, 0, &state);
    on_mouse(cv::EVENT_MBUTTONDOWN, 10, 10, 0, &state);

    EXPECT_FALSE(state.save_requested);
    EXPECT_EQ(state.click_pos, cv::Point(0, 0));
}


class DrawAndSaveTest : public ::testing::Test {
protected:
    void SetUp() override {
        save_dir = fs::temp_directory_path() / ("wc_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(save_dir);
    }
    void TearDown() override { fs::remove_all(save_dir); }
    fs::path save_dir;
};

TEST_F(DrawAndSaveTest, CreatesFileOnDisk)
{
    std::string path = draw_and_save(make_grey_frame(), {50, 50}, save_dir.string());
    ASSERT_FALSE(path.empty());
    EXPECT_TRUE(fs::exists(path));
}

TEST_F(DrawAndSaveTest, SavedFileHasPngExtension)
{
    std::string path = draw_and_save(make_grey_frame(), {10, 10}, save_dir.string());
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(fs::path(path).extension(), ".png");
}

TEST(LKTracker, InactiveBeforeInit)
{
    LKTracker t;
    EXPECT_FALSE(t.is_active());
}

TEST(LKTracker, ActiveAfterInit)
{
    LKTracker t;
    t.init(make_grey_frame(), {50, 50});
    EXPECT_TRUE(t.is_active());
}

TEST(LKTracker, ResetMakesInactive)
{
    LKTracker t;
    t.init(make_grey_frame(), {50, 50});
    t.reset();
    EXPECT_FALSE(t.is_active());
}

TEST(LKTracker, UpdateReturnsTrueOnIdenticalFrames)
{
    // LK optical flow needs image gradients, use a textured frame.
    // On two identical frames the tracked point should stay put (±2 px).
    LKTracker t;
    cv::Mat frame = make_textured_frame();
    t.init(frame, {50, 50});

    cv::Point out;
    bool ok = t.update(frame, out);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(out.x, 50, 2);
    EXPECT_NEAR(out.y, 50, 2);
}

TEST(LKTracker, ReinitRestartTracking)
{
    LKTracker t;
    cv::Mat frame = make_textured_frame();
    t.init(frame, {10, 10});
    t.reset();
    t.init(frame, {80, 80});  // restart at a different point
    EXPECT_TRUE(t.is_active());

    cv::Point out;
    EXPECT_TRUE(t.update(frame, out));
    EXPECT_NEAR(out.x, 80, 2);
    EXPECT_NEAR(out.y, 80, 2);
}

TEST(LKTracker, FailsOnUniformFrame)
{
    // A uniform frame has no gradients, LK correctly reports tracking lost.
    LKTracker t;
    cv::Mat frame = make_grey_frame();
    t.init(frame, {50, 50});

    cv::Point out;
    EXPECT_FALSE(t.update(frame, out));
    EXPECT_FALSE(t.is_active());
}

// MockCamera + MockTracker + no-op DisplaySink

TEST(Run, ExitsOnLowercaseQ)
{
    NiceMock<MockCamera>  cam;     setup_open_camera(cam);
    NiceMock<MockTracker> tracker; setup_inactive_tracker(tracker);
    DisplaySink display = make_noop_sink({'q'});
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_q", 10), 0);
}

TEST(Run, ExitsOnUppercaseQ)
{
    NiceMock<MockCamera>  cam;     setup_open_camera(cam);
    NiceMock<MockTracker> tracker; setup_inactive_tracker(tracker);
    DisplaySink display = make_noop_sink({'Q'});
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_Q", 10), 0);
}

TEST(Run, ExitsOnEscKey)
{
    NiceMock<MockCamera>  cam;     setup_open_camera(cam);
    NiceMock<MockTracker> tracker; setup_inactive_tracker(tracker);
    DisplaySink display = make_noop_sink({27});
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_esc", 10), 0);
}

// click initialises tracker and saves file

TEST(Run, ClickInitialisesTrackerAndSavesFile)
{
    fs::path save_dir = fs::temp_directory_path() / ("wc_click_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(save_dir);

    NiceMock<MockCamera>  cam; setup_open_camera(cam);
    NiceMock<MockTracker> tracker;
    ON_CALL(tracker, is_active()).WillByDefault(Return(false));

    // Expect init() to be called exactly once with any frame and the click point.
    EXPECT_CALL(tracker, init(_, cv::Point(50, 50))).Times(1);

    AppState* captured_state = nullptr;
    int show_calls = 0;
    DisplaySink display{
        [&](const std::string&, AppState* s) { captured_state = s; },
        [&](const std::string&, const cv::Mat&) -> int {
            // First call: inject click, keep looping. Second call: quit.
            if (show_calls++ == 0 && captured_state) {
                captured_state->save_requested = true;
                captured_state->click_pos      = {50, 50};
                return -1;
            }
            return 'q';
        },
        [](const std::string&, const cv::Mat&, int) {}
    };

    EXPECT_EQ(run(cam, tracker, display, save_dir.string(), 10), 0);

    bool found = false;
    for (const auto& e : fs::directory_iterator(save_dir))
        if (e.path().extension() == ".png") { found = true; break; }
    EXPECT_TRUE(found) << "No PNG saved in " << save_dir;

    fs::remove_all(save_dir);
}

// active tracker: update() is called each frame

TEST(Run, ActiveTrackerUpdatedEachFrame)
{
    NiceMock<MockCamera>  cam; setup_open_camera(cam);
    NiceMock<MockTracker> tracker;

    // Tracker is active and update() always succeeds at (40,40).
    ON_CALL(tracker, is_active()).WillByDefault(Return(true));
    ON_CALL(tracker, update(_, _)).WillByDefault(DoAll(
        SetArgReferee<1>(cv::Point(40, 40)),
        Return(true)
    ));

    // show_frame is called twice (two frames) before returning 'q'.
    EXPECT_CALL(tracker, update(_, _)).Times(2);

    DisplaySink display = make_noop_sink({-1, 'q'});
    run(cam, tracker, display, "/tmp/wc_track", 10);
}

// lost tracking calls reset()

TEST(Run, LostTrackingCallsReset)
{
    NiceMock<MockCamera>  cam; setup_open_camera(cam);
    NiceMock<MockTracker> tracker;

    ON_CALL(tracker, is_active()).WillByDefault(Return(true));
    // update() fails → tracker.reset() should be called.
    ON_CALL(tracker, update(_, _)).WillByDefault(Return(false));
    EXPECT_CALL(tracker, reset()).Times(::testing::AtLeast(1));

    DisplaySink display = make_noop_sink({'q'});
    run(cam, tracker, display, "/tmp/wc_lost", 10);
}

// resilience tests

TEST(Run, EmptyFrameDoesNotCrash)
{
    NiceMock<MockCamera>  cam;
    NiceMock<MockTracker> tracker; setup_inactive_tracker(tracker);
    ON_CALL(cam, isOpened()).WillByDefault(Return(true));
    ON_CALL(cam, release()).WillByDefault(Return());

    int call_count = 0;
    ON_CALL(cam, read(_)).WillByDefault(DoAll(
        Invoke([&](cv::Mat& f) { f = (++call_count > 3) ? make_grey_frame() : cv::Mat{}; }),
        Return(true)
    ));

    DisplaySink display = make_noop_sink({'q'});
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_empty", 3), 0);
}

TEST(Run, ReadFailureDoesNotCrash)
{
    NiceMock<MockCamera>  cam;
    NiceMock<MockTracker> tracker; setup_inactive_tracker(tracker);
    ON_CALL(cam, isOpened()).WillByDefault(Return(true));
    ON_CALL(cam, release()).WillByDefault(Return());

    int call_count = 0;
    ON_CALL(cam, read(_)).WillByDefault(DoAll(
        Invoke([&](cv::Mat& f) { if (call_count++ >= 3) f = make_grey_frame(); }),
        Invoke([&](cv::Mat&) -> bool { return call_count > 3; })
    ));

    DisplaySink display = make_noop_sink({'q'});
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_fail", 4), 0);
}

TEST(Run, ReturnsErrorWhenCameraNotOpen)
{
    NiceMock<MockCamera>  cam;
    NiceMock<MockTracker> tracker;
    ON_CALL(cam, isOpened()).WillByDefault(Return(false));

    DisplaySink display = make_noop_sink();
    EXPECT_EQ(run(cam, tracker, display, "/tmp/wc_noopen", 1), 1);
}
