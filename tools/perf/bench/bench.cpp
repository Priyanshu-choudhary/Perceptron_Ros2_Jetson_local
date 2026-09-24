// Single-thread kernels shaped like the nav stack's hot loops.
// Same source, same -O2, run on the laptop and on the Jetson Nano; the ratio of
// times is the per-core speed factor used to translate measured laptop load.
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static volatile double sink;

template <class F> double best_of(int reps, F f) {
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    f();
    double s = std::chrono::duration<double>(Clock::now() - t0).count();
    if (s < best) best = s;
  }
  return best;
}

// 1. costmap obstacle layer: raytrace 720 beams (clearing) + mark endpoints
double k_raytrace() {
  const int W = 400;  // 20 m @ 0.05
  std::vector<uint8_t> g(W * W, 0);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> rr(0.5, 8.0);
  std::vector<double> ranges(720);
  for (auto &r : ranges) r = rr(rng);
  return best_of(5, [&] {
    for (int it = 0; it < 60; ++it) {
      int x0 = W / 2, y0 = W / 2;
      for (int b = 0; b < 720; ++b) {
        double a = b * 2 * M_PI / 720;
        int x1 = x0 + int(ranges[b] * 20 * std::cos(a)), y1 = y0 + int(ranges[b] * 20 * std::sin(a));
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, x = x0, y = y0;
        while (x != x1 || y != y1) {
          g[y * W + x] = 0;
          int e2 = 2 * err;
          if (e2 >= dy) { err += dy; x += sx; }
          if (e2 <= dx) { err += dx; y += sy; }
        }
        g[y1 * W + x1] = 254;
      }
    }
    sink = g[123];
  });
}

// 2. inflation layer: BFS distance propagation over a 480x480 grid
double k_inflation() {
  const int W = 480;
  std::vector<uint8_t> cost(W * W, 0);
  std::vector<float> dist(W * W);
  std::mt19937 rng(2);
  std::vector<int> seeds;
  for (int i = 0; i < 3000; ++i) seeds.push_back(rng() % (W * W));
  return best_of(5, [&] {
    for (int it = 0; it < 4; ++it) {
      std::fill(dist.begin(), dist.end(), 1e9f);
      std::deque<int> q;
      for (int s : seeds) { dist[s] = 0; q.push_back(s); }
      while (!q.empty()) {
        int c = q.front(); q.pop_front();
        int cx = c % W, cy = c / W;
        float d = dist[c];
        if (d > 8) continue;  // 0.4 m radius
        const int nb[4] = {c - 1, c + 1, c - W, c + W};
        const bool ok[4] = {cx > 0, cx < W - 1, cy > 0, cy < W - 1};
        for (int k = 0; k < 4; ++k)
          if (ok[k] && dist[nb[k]] > d + 1) { dist[nb[k]] = d + 1; q.push_back(nb[k]); }
      }
      for (int i = 0; i < W * W; ++i)
        cost[i] = dist[i] == 0 ? 254 : (dist[i] < 7 ? 253 : uint8_t(252 * std::exp(-3.0 * 0.05 * (dist[i] - 6.6))));
    }
    sink = cost[777];
  });
}

// 3. DWB: 800 trajectories x 34 steps, costmap lookups + path-distance scoring
double k_dwb() {
  const int W = 80;
  std::vector<uint8_t> cm(W * W);
  std::mt19937 rng(3);
  for (auto &c : cm) c = rng() % 200;
  return best_of(5, [&] {
    double best = 1e18;
    for (int cyc = 0; cyc < 20; ++cyc)
      for (int vx = 0; vx < 20; ++vx)
        for (int vt = 0; vt < 40; ++vt) {
          double v = 0.35 * vx / 19.0, w = -1.0 + 2.0 * vt / 39.0;
          double x = 0, y = 0, th = 0, score = 0;
          for (int s = 0; s < 34; ++s) {
            x += v * std::cos(th) * 0.05; y += v * std::sin(th) * 0.05; th += w * 0.05;
            int mx = W / 2 + int(x / 0.05), my = W / 2 + int(y / 0.05);
            score += cm[(my % W + W) % W * W + (mx % W + W) % W];
            score += 32 * std::hypot(x - 1.0, y) + 24 * std::fabs(std::atan2(y, x - 1.0) - th);
          }
          if (score < best) best = score;
        }
    sink = best;
  });
}

// 4. AMCL likelihood field: 2000 particles x 240 beams
double k_amcl() {
  const int W = 241, H = 219;
  std::vector<float> lf(W * H);
  std::mt19937 rng(4);
  for (auto &v : lf) v = (rng() % 1000) / 1000.0f;
  std::vector<double> px(2000), py(2000), pt(2000), ranges(240);
  for (int i = 0; i < 2000; ++i) { px[i] = 3 + (rng() % 100) / 50.0; py[i] = 3 + (rng() % 100) / 50.0; pt[i] = (rng() % 628) / 100.0; }
  for (auto &r : ranges) r = 0.5 + (rng() % 700) / 100.0;
  return best_of(5, [&] {
    double tot = 0;
    for (int it = 0; it < 10; ++it)
      for (int p = 0; p < 2000; ++p) {
        double w = 1.0;
        for (int b = 0; b < 240; ++b) {
          double a = pt[p] + b * 2 * M_PI / 240;
          double hx = px[p] + ranges[b] * std::cos(a), hy = py[p] + ranges[b] * std::sin(a);
          int mx = int(hx / 0.05), my = int(hy / 0.05);
          double z = (mx >= 0 && mx < W && my >= 0 && my < H) ? lf[my * W + mx] : 2.0;
          double pz = 0.5 * std::exp(-(z * z) / 0.08) + 0.5 / 12.0;
          w += pz * pz * pz;
        }
        tot += w;
      }
    sink = tot;
  });
}

// 5. executor wake-ups: two threads ping-pong through a condition variable
//    (the futex round trip every ROS callback pays at least once)
double k_wakeup() {
  return best_of(3, [&] {
    std::mutex m;
    std::condition_variable cv;
    int turn = 0;
    const int N = 20000;
    std::thread t([&] {
      for (int i = 0; i < N; ++i) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return turn == 1; });
        turn = 0;
        cv.notify_one();
      }
    });
    for (int i = 0; i < N; ++i) {
      std::unique_lock<std::mutex> l(m);
      turn = 1;
      cv.notify_one();
      cv.wait(l, [&] { return turn == 0; });
    }
    t.join();
  });
}

// 6. message serialisation-like memcpy of a 900 kB costmap_raw, 200 times
double k_memcpy() {
  std::vector<uint8_t> a(921600, 7), b(921600);
  return best_of(5, [&] {
    for (int i = 0; i < 200; ++i) { std::memcpy(b.data(), a.data(), a.size()); a[i] = b[i + 1]; }
    sink = b[5];
  });
}

int main() {
  std::printf("{\"raytrace\": %.5f, \"inflation\": %.5f, \"dwb\": %.5f, \"amcl\": %.5f, \"wakeup\": %.5f, \"memcpy\": %.5f}\n",
              k_raytrace(), k_inflation(), k_dwb(), k_amcl(), k_wakeup(), k_memcpy());
}
