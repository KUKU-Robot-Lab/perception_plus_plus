from enum import IntEnum


class TrackingState(IntEnum):
    INITIALIZING = 0
    TRACKING = 1
    LOST = 2
    REINITIALIZING = 3

