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
# Pre-place Cutie weights where Cutie actually loads them (Cutie/weights), so a
# fresh container does not re-download ~180 MB on every start. (The adapter's
# _link_models symlinks them one dir too deep, at Cutie/cutie/weights.)
RUN cd /workspace/perception_plus_plus && mkdir -p \
      external/foundationpose_plus_plus/Cutie/weights && \
    cp models/cutie/cutie-base-mega.pth models/cutie/coco_lvis_h18_itermask.pth \
      external/foundationpose_plus_plus/Cutie/weights/
# Pre-cache the torchvision ResNet backbones Cutie pulls on first init, so a
# fresh container starts in seconds instead of downloading ~150 MB.
RUN mkdir -p /home/perception/.cache/torch/hub/checkpoints && \
    wget -q https://download.pytorch.org/models/resnet50-19c8e357.pth \
      -O /home/perception/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth && \
    wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
      -O /home/perception/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth && \
    chown -R perception:perception /home/perception/.cache
COPY scripts/live_fp_demo.py /workspace/perception_plus_plus/scripts/live_fp_demo.py
USER perception
CMD ["bash"]
