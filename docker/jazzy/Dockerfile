FROM ros:jazzy-ros-base-noble

ARG ROS_DISTRO=jazzy
ARG TARGET_OS=ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libeigen3-dev libgl1 libglib2.0-0 \
    libboost-system-dev libboost-program-options-dev pybind11-dev \
    python3-colcon-common-extensions python3-dev python3-pip wget \
    ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf2-ros && rm -rf /var/lib/apt/lists/*
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
      -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-4 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=/usr/local/cuda-12.4/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY docker/jazzy/requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.lock
RUN useradd --create-home --uid 1000 --shell /bin/bash perception
WORKDIR /workspace/perception_plus_plus
COPY pyproject.toml README.md ./
COPY perception_plus_plus_core perception_plus_plus_core
COPY ros_ws/src ros_ws/src
COPY external/foundationpose_plus_plus external/foundationpose_plus_plus
COPY scripts/build_fpplusplus.sh scripts/build_fpplusplus.sh
RUN python3 -m pip install --break-system-packages --no-cache-dir .
RUN bash scripts/build_fpplusplus.sh
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build \
    --base-paths ros_ws/src --install-base /opt/perception_plus_plus --merge-install
COPY . .
RUN chown -R perception:perception /workspace/perception_plus_plus
USER perception
CMD ["bash"]
