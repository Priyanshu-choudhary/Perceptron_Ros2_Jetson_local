import zmq, time, pickle, collections, sys
secs = float(sys.argv[1])
c = zmq.Context(); s = c.socket(zmq.SUB); s.setsockopt(zmq.SUBSCRIBE, b""); s.setsockopt(zmq.RCVTIMEO, 500)
s.setsockopt(zmq.RCVHWM, 10000); s.connect("tcp://192.168.1.7:5555")
time.sleep(1.0)
frames = []; t0 = time.time()
while time.time() - t0 < secs:
    try:
        f = s.recv_multipart(); frames.append((time.time() - t0, f[0], f[1]))
    except zmq.Again:
        pass
cnt = collections.Counter(t for _, t, _ in frames)
print({k.decode(): round(v / secs, 1) for k, v in cnt.items()}, len(frames))
pickle.dump(frames, open(sys.argv[2], 'wb'))
