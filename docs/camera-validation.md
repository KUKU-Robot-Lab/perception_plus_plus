# D435i Camera Validation

Start the RealSense driver and tracker with the validated profile:

```bash
ros2 launch perception_plus_plus_ros realsense_cup_tracking.launch.py \
  project_root:="$PWD"
```

This selects synchronized, aligned 640x480 RGB-D at 30 Hz. The tracker
subscribes with the ROS sensor-data QoS profile.

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
