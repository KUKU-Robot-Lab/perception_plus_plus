# D435i Camera Validation

Start the RealSense driver with aligned depth enabled, then start the tracker:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
ros2 launch perception_plus_plus_ros cup_tracking.launch.py
```

Before accepting results, verify:

- RGB and camera info use the same optical frame.
- Depth is `aligned_depth_to_color`, not the raw depth optical frame.
- No pose or TF appears while status is `LOST`.
- Initial appearance reaches `TRACKING`.
- Partial occlusion remains stable.
- Full occlusion or departure reaches `LOST`.
- Reappearance passes `REINITIALIZING` before `TRACKING`.
- VRAM does not grow during at least one hour.

Record FPS, end-to-end callback latency, initialization latency, peak/steady
VRAM, and recovery duration. Those measurements become the first regression
baseline; this repository intentionally does not invent target values before
hardware measurement.

