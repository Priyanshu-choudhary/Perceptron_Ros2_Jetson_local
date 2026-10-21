# Short-horizon prediction of one tracked object (roadmap Stage 7). Frame and the time the
# horizons count from are the enclosing array's header.

uint8 STATIC=0        # gated as not moving: stays where it is
uint8 CV=1            # constant velocity
uint8 CTRV=2          # constant turn rate and velocity (vehicles with a measured turn)

uint32 id                                     # the track id (Stage 6)
string class_id
uint8 model
float32 speed                                 # m/s actually used (after shrinkage and caps)
float32 yaw_rate                              # rad/s used by CTRV, else 0
float32 radius                                # footprint radius for the costmap, m
float32[] horizons                            # s after header.stamp
geometry_msgs/PoseWithCovariance[] poses      # predicted centre at each horizon; covariance
                                              # [0,1,6,7] is the 2D position uncertainty
geometry_msgs/Point[] path                    # dense centre line from now to the last horizon
float32 path_dt                               # s between path points
