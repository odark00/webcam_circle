# webcam_circle

Real-time webcam viewer that lets you click to mark a point, track it across frames using optical flow, and save captures.

## Features

- Live webcam feed displayed in an GUI window
- Left-click to select a point, a green circle marks it and the frame is saved
- The circle follows the point across subsequent frames (Lucas-Kanade optical flow)
- Tracker algorithm is swappable via a `ITracker` interface
- Camera source is exchangeble via `ICamera`

---

## Requirements

| Dependency | Version | Install |
|---|---|---|
| GCC / Clang | C++17 | `sudo apt install build-essential` |
| CMake | ≥ 3.16 | `apt install cmake` |
| OpenCV | ≥ 4.x | `apt install libopencv-dev` |
| GoogleTest | v1.14 | `apt-get install libgtest-dev libgmock-dev -y` |

---

## Build

```bash
cmake .. -DBUILD_TESTS=ON

make && ./tests
```

## Run

```bash
./build/webcam_circle          # default camera (/dev/video0)
./build/webcam_circle 1        # use /dev/video1
```

### Controls

| Input | Action |
|---|---|
| **Left-click** | Mark point, start/restart tracking, save image |
| **Q** or **Esc** | Quit |

Saved images are written to `captures/YYYYMMDD_HHMMSS_mmm.png` in the working directory.
