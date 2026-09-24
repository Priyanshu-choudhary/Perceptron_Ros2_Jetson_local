# Jetson-side scripts

Code that runs **on the Jetson Nano**, not on the ROS host. Nothing here is a
ROS package and `colcon` never sees it — copy it across by hand:

```bash
scp jetson/jetson_path_overlay.py jetson@192.168.1.7:/home/jetson/
```

It is version-controlled here anyway, because the alternative is a file that
exists only on an SD card.

## `jetson_robot_bridge.py` — serial / camera bridge

LiDAR + STM32 over serial and ArUco corners from the USB camera, out over ZMQ
(PUB :5555, PULL :5556). The copy here is the one running on the robot.

Camera modes:

| flags | camera |
|---|---|
| (none) | always on, 5 Hz detection (`robot start ... aruco=true`) |
| `--aruco-on-demand` | closed, OpenCV not even imported, until `{'aruco_enable': true, 'hold': s, 'rate': Hz}` arrives on :5556; closes again when `hold` runs out. `robot` uses this by default; the dock tool resends it every second while docking. |
| `--no-aruco` | no camera thread at all |

`aruco_enable` deliberately does not refresh the motor watchdog, so a camera
keepalive can never keep a robot moving whose `/cmd_vel` source died.

## `jetson_path_overlay.py` — AR path overlay renderer

Draws the Nav2 plan onto the live camera and streams the result back as H.265.

```
host  path_overlay_node.py  --ZMQ PUSH :5558-->  jetson_path_overlay.py
                                                 /dev/video0 -> draw -> H.265
                                                        |
                                                   RTP/UDP :5000
                                                        v
                                                    operator
```

### Running it

The camera can only be held by one process, so free it first:

```bash
python3 jetson_robot_bridge.py --no-aruco
```

then, in a second shell:

```bash
python3 jetson_path_overlay.py --host-ip 192.168.1.5
```

On the operator machine:

```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H265, payload=96" ! rtph265depay ! h265parse ! avdec_h265 ! autovideosink sync=false
```

And on the ROS host, alongside `navigation.launch.py`:

```bash
ros2 launch perceptron_navigation path_overlay.launch.py
```

### What it knows, and what it deliberately does not

This process receives a list of shapes in pixel coordinates and draws them. It
has no idea what a path, a transform or a costmap is. Every piece of geometry —
the TF chain, the intrinsics, the lens model, which Nav2 topic is which — stays
on the host in `path_overlay_node.py`.

That split is the point. Projection bugs stay debuggable on a machine with a
debugger on it, and this file does not need redeploying when the robot changes.

### Wire format

msgpack over a ZMQ PUSH/PULL pair; this process **binds** PULL on 5558, matching
how `jetson_robot_bridge.py` binds 5555 and 5556. Colours are **RGB** so they
read the same as the host's ROS parameters; the renderer converts to BGR.

```python
{'v': 1, 'seq': int, 'ref': [1280, 720],      # ref = resolution (u,v) are in
 'polygons':  [{'pts': [[u,v],...], 'rgb': [r,g,b], 'alpha': 0.3}],
 'polylines': [{'pts': [[u,v],...], 'rgb': [r,g,b], 'w': 3}],
 'circles':   [{'c': [u,v], 'r': 11, 'rgb': [r,g,b], 'w': 2}],
 'labels':    [{'p': [u,v], 't': 'GOAL', 'rgb': [r,g,b], 's': 0.55}],
 'hud':       ['plan 6.43 m  120 pts']}
```

`ref` is what makes `--width`/`--height` a purely local decision: the host always
projects at the calibrated 1280x720 and this end rescales.

### Clocks

**No timestamp is ever compared across the two machines.** The host's clock and
the Jetson's are independent and undisciplined — `jetson_bridge_node.py` carries
an entire `JetsonClock` class because of it. An overlay's age is measured here as
`time.monotonic()` since *this* process received it. Getting this wrong would
make overlays look permanently fresh or permanently stale depending on drift,
and would be a slow bug to find.

Past `--max-age` (0.7 s) the overlay is dropped and clean video keeps flowing. A
path frozen where the robot *used to be* going is worse than no path at all.

### Useful flags

| flag | why |
| --- | --- |
| `--test-src` | `videotestsrc` instead of the camera. Proves the encode and ZMQ path when the camera is busy. |
| `--print-pipelines` | Print both GStreamer pipelines and exit. |
| `--sw-decode` | CPU `jpegdec` instead of the default hardware decoder. ~30x more CPU; for a board where `nvv4l2decoder` misbehaves. |
| `--sw-encode` | `x265enc` fallback if the hardware encoder will not negotiate. |
| `--no-hud` | Drop the text block. |

### Measured on the hardware (JetPack R32.7.6, OpenCV 4.5.5)

* The camera offers MJPEG and H.264 at **1280x720 and 1920x1080, 30 fps only** —
  there is no 15 fps mode, which is why `videorate` is in the pipeline rather
  than just asking v4l2 for 15.
* Sustained **15.0 fps** out with every frame overlaid, on all three capture
  modes (hardware decode, `--sw-decode`, `--test-src`).
* `videorate` sits **before** the decoder, on the still-compressed JPEG stream:
  dropping a JPEG is free, decoding one and then discarding it is not. CPU over
  10 s windows, above a 2.4% idle baseline on the 4-core Nano:

  | capture path | CPU |
  | --- | --- |
  | `jpegdec`, videorate after | 15.7% |
  | `jpegdec`, videorate before | 11.9% |
  | `nvv4l2decoder`, videorate before | **0.5%** |

  Hence hardware decode is the default — it is ~30x cheaper than `jpegdec` on a
  board that is also running the lidar, odometry and IMU bridge.
* `config-interval=1` on `rtph265pay` is the one deliberate change from the
  hand-rolled pipeline: it repeats SPS/PPS so a receiver started *after* the
  stream still decodes, instead of showing green mush until the next IDR.

### Gotchas found the hard way

* `pkill -f jetson_path_overlay` **matches the ssh command running it** and kills
  your own shell. Use `pgrep -f "[j]etson_path" | xargs -r kill`.
* Under `nohup`/`systemd`/`ssh host cmd`, stdout is a pipe and Python
  block-buffers it. Status goes through `say()`, which flushes; use `python3 -u`
  if you add prints.
* `videorate` **after** `nvvidconv` with an explicit framerate capsfilter fails
  to negotiate (`-4`): the second capsfilter drops the format `nvvidconv`
  settled on. `videorate max-rate=15` there does work, but still decodes every
  frame. Putting `videorate` before the decoder avoids both problems.
* `nvv4l2decoder` emits `video/x-raw(memory:NVMM)`, which `videorate` and other
  CPU elements cannot touch at all — `nvvidconv` has to come first.
* `Device '/dev/video0' is busy` means something else holds the camera —
  usually `jetson_robot_bridge.py` started without `--no-aruco`, or a previous
  run that outlived its ssh session.
