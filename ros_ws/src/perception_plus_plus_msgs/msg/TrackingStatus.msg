uint8 INITIALIZING=0
uint8 TRACKING=1
uint8 LOST=2
uint8 REINITIALIZING=3

std_msgs/Header header
uint8 state
builtin_interfaces/Time last_valid_pose_stamp
string failure_reason
string failure_detail
uint32 consecutive_valid
uint32 consecutive_invalid
bool fatal

