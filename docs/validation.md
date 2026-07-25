# Validation Contract

`scripts/pre_camera_check.py` reports `PASS`, `FAIL`, or `SKIP` for each
capability. A required `SKIP` keeps the system `NOT_READY`; optional tooling
such as the Docker CLI may be skipped.

The model manifest intentionally ships with all-zero digest templates because
the upstream project does not define one canonical redistribution set. This is
a fail-closed state, not a usable digest. A deployment owner must select
licensed weights, record their real SHA-256 values, and retain license notices.

Pre-camera readiness requires:

1. the pinned FP++ submodule;
2. every required model and the configured mesh;
3. the CPU test suite;
4. ROS message/node build and tests for the target distribution;
5. CUDA, PyTorch, FP++ extension imports, and one recorded initialization;
6. deterministic replay without fatal state;
7. a successful container smoke test for each distribution being released.

The framework check cannot replace GPU inference. Run
`scripts/fpplusplus_smoke.py --npz <initial-frame.npz> --mesh <mesh>` inside
each target container before attaching the D435i.

