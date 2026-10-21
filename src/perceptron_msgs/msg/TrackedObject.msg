# One tracked object (roadmap Stage 6). Frame and stamp come from the enclosing array.

uint8 TENTATIVE=0     # seen, not yet confirmed: do not plan around it yet
uint8 CONFIRMED=1     # updated by a detection this frame
uint8 COASTING=2      # confirmed, not seen this frame: position is a prediction

uint32 id                                     # tracker-generated, stable for the object's life
string class_id                               # majority vote of the detections it absorbed
float32 class_confidence                      # share of its detections that voted for class_id
uint8 state
bool moving                                   # speed clearly above zero (see the tracker docs)
geometry_msgs/PoseWithCovariance pose         # position in the array's frame (odom)
geometry_msgs/TwistWithCovariance twist       # velocity in the same frame, m/s
geometry_msgs/Vector3 size                    # extent of its last lidar points, m
float32 age                                   # seconds since the first detection
uint32 hits                                   # detections absorbed
uint32 misses                                 # consecutive frames without one
string source_detection_id                    # id of the detection used this frame ("" if none);
                                              # passed through for evaluation, never used to track
float64[16] state_covariance                  # full [x, y, vx, vy] covariance, row-major: the
                                              # position-velocity cross terms that pose/twist
                                              # covariance leave out (needed for prediction)
