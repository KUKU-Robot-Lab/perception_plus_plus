# Live-demo image: adds camera I/O, a modern YOLO, and a GUI OpenCV on top of
# the validated perception-plus-plus:humble base. The base FP++ env pins an old
# ultralytics with a broken wandb integration; upgrading it (and dropping wandb)
# fixes detection without touching the FP core.
#
# Build (context = repo root):
#   docker build -f docker/demo/Dockerfile -t perception-plus-plus:humble-demo .
FROM perception-plus-plus:humble
USER root
RUN pip uninstall -y wandb opencv-python-headless >/dev/null 2>&1 || true; \
    pip install --no-cache-dir -U ultralytics && \
    pip install --no-cache-dir opencv-python pyrealsense2
COPY scripts/live_fp_demo.py /workspace/perception_plus_plus/scripts/live_fp_demo.py
USER perception
CMD ["bash"]
