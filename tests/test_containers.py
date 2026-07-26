from pathlib import Path


def test_distribution_specific_containers_share_source_contract():
    humble = Path("docker/humble/Dockerfile").read_text()
    jazzy = Path("docker/jazzy/Dockerfile").read_text()
    assert "ubuntu22.04" in humble and "ROS_DISTRO=humble" in humble
    assert "ubuntu24.04" in jazzy and "ROS_DISTRO=jazzy" in jazzy
    for text in (humble, jazzy):
        assert "ros_ws/src" in text
        assert "USER perception" in text
        assert "NVIDIA_VISIBLE_DEVICES" in text
        assert "--ignore-installed" in text
        assert "--uid 10001" in text
        assert "--uid 1000 " not in text


def test_ros_sources_are_copied_after_the_expensive_build_stages():
    for dockerfile in ("docker/humble/Dockerfile", "docker/jazzy/Dockerfile"):
        lines = Path(dockerfile).read_text().splitlines()
        copy_ros = next(i for i, line in enumerate(lines)
                        if line.startswith("COPY ros_ws/src"))
        extensions = next(i for i, line in enumerate(lines)
                          if "build_fpplusplus.sh extensions" in line)
        assert copy_ros > extensions


def test_jazzy_uses_first_cuda_toolkit_available_for_noble():
    jazzy = Path("docker/jazzy/Dockerfile").read_text()
    assert "cuda-toolkit-12-5" in jazzy
    assert "CUDA_HOME=/usr/local/cuda-12.5" in jazzy
    assert "cuda-toolkit-12-4" not in jazzy


def test_lock_files_are_distribution_specific():
    humble = Path("docker/humble/requirements.lock").read_text()
    jazzy = Path("docker/jazzy/requirements.lock").read_text()
    assert humble != jazzy
    assert "torch==" in humble and "torch==" in jazzy
    assert "gdown==" in humble and "gdown==" in jazzy


def test_locks_keep_numpy_compatible_with_apt_built_ros_extensions():
    # cv_bridge and the other ROS binaries come from apt and are compiled
    # against the distribution's NumPy 1.x. Installing NumPy 2 over them breaks
    # every imgmsg_to_cv2 call at runtime.
    for lock in ("docker/humble/requirements.lock", "docker/jazzy/requirements.lock"):
        pins = dict(
            line.split("==", 1) for line in Path(lock).read_text().splitlines() if line
        )
        assert pins["numpy"].startswith("1."), f"{lock} pins numpy {pins['numpy']}"


def test_docker_context_excludes_external_model_bytes():
    ignore = Path(".dockerignore").read_text().splitlines()
    assert "models/" in ignore
    assert "reports/" in ignore


def test_container_smoke_mounts_and_verifies_model_lock():
    smoke = Path("scripts/container_smoke.sh").read_text()
    assert 'models:/workspace/perception_plus_plus/models:ro' in smoke
    assert "bootstrap_models.py --verify-only" in smoke


def test_container_smoke_exercises_cv_bridge_against_installed_numpy():
    smoke = Path("scripts/container_smoke.sh").read_text()
    assert "from cv_bridge import CvBridge" in smoke
    assert "imgmsg_to_cv2" in smoke


def test_dockerfiles_constrain_every_pip_stage_to_the_lock():
    for distro in ("humble", "jazzy"):
        lines = Path(f"docker/{distro}/Dockerfile").read_text().splitlines()
        constraint = next(i for i, line in enumerate(lines)
                          if line.startswith("ENV PIP_CONSTRAINT=/tmp/requirements.lock"))
        stages = [i for i, line in enumerate(lines)
                  if "pip install" in line or "build_fpplusplus.sh" in line]
        assert stages and constraint < min(stages)


def test_fpplusplus_build_uses_curated_runtime_dependencies_and_cacheable_stages():
    script = Path("scripts/build_fpplusplus.sh").read_text()
    requirements = Path("docker/foundationpose-runtime.requirements").read_text()
    assert "docker/foundationpose-runtime.requirements" in script
    assert "FoundationPose/requirements.txt" not in script
    assert '--no-deps -e "$upstream_dir/Cutie"' in script
    assert "open3d==0.18.0; python_version < '3.12'" in requirements
    assert "open3d==0.19.0; python_version >= '3.12'" in requirements
    for unwanted in ("cchardet", "g4f", "gradio", "jupyterlab", "wandb"):
        assert unwanted not in requirements.lower()
    for stage in ("nvdiffrast", "pytorch3d", "requirements", "extensions"):
        assert f"{stage})" in script
    for dockerfile in ("docker/humble/Dockerfile", "docker/jazzy/Dockerfile"):
        text = Path(dockerfile).read_text()
        assert "docker/foundationpose-runtime.requirements" in text
        for stage in ("nvdiffrast", "pytorch3d", "requirements", "extensions"):
            assert f"build_fpplusplus.sh {stage}" in text
