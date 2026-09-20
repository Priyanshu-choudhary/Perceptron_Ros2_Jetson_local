Projects `/plan` and `/local_plan` into the front camera's pixels and draws them on the live video, so the operator sees where the robot intends to go on the floor of the actual room rather than on a map beside it.

## Architecture: host projects, Jetson draws

The camera never joins the ROS graph — an uncompressed 1280x720 stream is ~41 MB/s, and `perceptron_robot_description/README.md` records what that does to node discovery on a 4 GB WSL host. The frame cannot come to the geometry, so the geometry goes to the frame: a decimated path is ~2.8 kB, about 41 kB/s at 15 Hz.

```
host  path_overlay_node.py  --ZMQ PUSH :5558-->  jetson_path_overlay.py
                                                 /dev/video0 -> draw -> H.265
                                                        |
                                                   RTP/UDP :5000 -> operator
```

`path_overlay_node.py` owns TF, the intrinsics and the Nav2 topics. `jetson/jetson_path_overlay.py` receives a list of shapes in pixel coordinates and draws them, knowing nothing about ROS. Projection bugs stay debuggable on a machine with a debugger on it, and the Jetson file needs no redeploy when the robot changes.

**Nothing in this path steers the robot.** If the link dies the operator loses a drawing; that is the whole failure mode.

Port 5558 is new; the Jetson binds PULL exactly as `jetson_robot_bridge.py` binds 5555 and 5556, so no bridge change is needed. The camera is freed with the bridge's existing `--no-aruco` flag (verified it really does gate the thread that opens `/dev/video0`).

## Reviewer notes

**The cull is two-stage, and `Z > 0` is not sufficient.** The lens is ~115° horizontal (fx ≈ 411 over 1280 px) and a 5-coefficient plumb-bob model is only valid inside the cone it was fitted in. With this calibration the radial polynomial stops being monotonic at r = 2.653 and folds past it: points *outside* the field of view come back with plausible pixel coordinates *inside* the image. Sweeping ground points across bearings, **37 land inside the frame** — e.g. 73° off axis projecting to (1252, 482). That would be a phantom stripe whipping across the picture whenever the robot turns and the path sweeps past the lens edge. So the mask is `Z > min_z` **and** `r <= 2.35` (66.9°), above the image corner (1.761) and below the fold. It also bounds projected pixel magnitude, which stops the int32 overflow an unculled point produces.

**No timestamp is compared across the two machines.** The host's clock and the Jetson's are independent and undisciplined — `JetsonClock` in `jetson_bridge_node.py` exists because of it. Overlay age is measured on the Jetson as `time.monotonic()` since it received the payload. The TF lookup asks for the *latest* transform rather than the frame's capture time, because the frame is on the far side of a UDP video link with no shared clock (`aruco_detector_node.py` already does the same). The cost is that the overlay swims slightly while the robot moves; the HUD says so past 0.35 s and the overlay is dropped past 0.7 s.

**`videorate` sits before the decoder** on the still-compressed JPEG stream. The camera offers 30 fps and nothing slower, so the halving has to happen somewhere, and dropping a JPEG is free while decoding one then discarding it is not. CPU over 10 s windows, above a 2.4% idle baseline on the 4-core Nano:

| capture path | CPU |
| --- | --- |
| `jpegdec`, videorate after | 15.7% |
| `jpegdec`, videorate before | 11.9% |
| `nvv4l2decoder`, videorate before | **0.5%** |

Hardware decode is therefore the default (~30x cheaper on a board also running the lidar, odometry and IMU bridge); `--sw-decode` is the fallback. Two GStreamer traps documented in the source: `videorate` *after* `nvvidconv` with an explicit framerate capsfilter fails to negotiate (`-4`), and `nvv4l2decoder` emits NVMM memory that CPU elements cannot touch at all.

## Testing

Verified end to end on the actual robot, not in simulation:

- 581 payloads over Wi-Fi, **0 dropped**; **75 of 75 frames overlaid**
- Steady **15.0 fps** on all three capture modes (hardware decode, `--sw-decode`, `--test-src`)
- 251 frames decoded back out of the live H.265 stream with the overlay burned in
- Overlay correctly ages out when the sender stops
- Projection checked against hand-computed geometry: horizon at v=377, floor visible from 0.535 m, 2/5/8 m at v=465/411/398, ribbon 193 px wide at 1 m and 47 px at 4 m
- `colcon build` clean; executable, launch file and config all install

The ribbon lying flat on the floor in the live stream is itself the check that the camera extrinsics (0.42 m, pitch −0.07027) are still true — a 1° pitch error puts the path ~9 cm off at 5 m and visibly floats or sinks it.

## Known limitations

- The overlay lags the picture by the video pipeline's latency while driving (see the clocks note above). Cosmetic, and surfaced on the HUD.
- Tested with synthetic paths pushed over the real link, not yet with a live Nav2 goal. The remaining proof is sending a 2D Nav Goal and watching the ribbon land on the floor being driven toward.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
