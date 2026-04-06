import ctypes
ctypes.CDLL("libcuda.so")

import numpy as np
import range_libc
import time

MAP_PATH = b"/home/racecar/racecar_ws/src/range_libc/maps/basement_hallways_5cm.png"
MAX_RANGE = 500.0
NUM_PARTICLES = 5000
NUM_ANGLES = 108
NUM_RUNS = 10

print("range_libc imported successfully")
print("USE_CUDA:", range_libc.SHOULD_USE_CUDA)

# Map loading
t0 = time.perf_counter()
omap = range_libc.PyOMap(MAP_PATH)
print(f"Map loaded: {omap.width()} x {omap.height()} in {(time.perf_counter()-t0)*1000:.1f}ms")

# RMGPU construction
t0 = time.perf_counter()
rmgpu = range_libc.PyRayMarchingGPU(omap, MAX_RANGE)
print(f"PyRayMarchingGPU initialized in {(time.perf_counter()-t0)*1000:.1f}ms\n")

# Test inputs
particles = np.array([[200.0, 200.0, a] for a in np.linspace(0, 2*np.pi, NUM_PARTICLES)], dtype=np.float32)
angles = np.linspace(-1.35, 1.35, NUM_ANGLES).astype(np.float32)
outs = np.zeros(NUM_PARTICLES * NUM_ANGLES, dtype=np.float32)

# Warmup
rmgpu.calc_range_repeat_angles(particles, angles, outs)

# Benchmark
times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    rmgpu.calc_range_repeat_angles(particles, angles, outs)
    times.append(time.perf_counter() - t0)

avg_ms = np.mean(times) * 1000
rays_per_sec = (NUM_PARTICLES * NUM_ANGLES) / np.mean(times)
print(f"Particles:   {NUM_PARTICLES}")
print(f"Beams:       {NUM_ANGLES}")
print(f"Avg time:    {avg_ms:.2f}ms")
print(f"Rays/sec:    {rays_per_sec:.0f}")
print(f"\nSample output (first particle):")
print(outs[:NUM_ANGLES].reshape(1, NUM_ANGLES))
