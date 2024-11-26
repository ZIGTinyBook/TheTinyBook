# Z-Ant

![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)

## Project Overview

**Zant** (Zig-Ant) is an open-source SDK designed to simplify deploying Neural Networks (NN) on microprocessors. Written in Zig, Zant prioritizes cross-compatibility and efficiency, providing tools to import, optimize, and deploy NNs seamlessly, tailored to specific hardware.

### Why Zant?

1. Many microcontrollers (e.g., ATMEGA, TI Sitara) lack robust deep learning libraries.
2. No open-source solution exists for end-to-end NN optimization and deployment.
3. Inspired by cutting-edge research (e.g., MIT Han Lab), we leverage state-of-the-art optimization techniques.
4. Collaborating with institutions like Politecnico di Milano to advance NN deployment on constrained devices.
5. Built for flexibility to adapt to new hardware without codebase changes.

### Key Features

- **Optimized Performance:** Supports quantization, pruning, and hardware acceleration (SIMD, GPU offloading).
- **Efficient Memory Usage:** Incorporates memory pooling, static allocation, and buffer optimization.
- **Cross-Platform Support:** Works on ARM Cortex-M, RISC-V, and more.
- **Ease of Integration:** Modular design with clear APIs, examples, and documentation.

### Use Cases

- **Real-Time Applications:** Object detection, anomaly detection, and predictive maintenance on edge devices.
- **IoT and Autonomous Systems:** Enable AI in IoT, drones, robots, and vehicles with constrained resources.

---

## Getting Started

### Prerequisites
1. Install the [latest Zig compiler](https://ziglang.org/learn/getting-started/).
2. Brush up your Zig skills with [Ziglings exercises](https://codeberg.org/ziglings/exercises).

### Run
Navigate to the project folder and execute:
```sh
zig build run
```

### Test
1. Add new test files to `build.zig/test_list` if not already listed.
2. Run:
   ```sh
   zig build
   zig build test_all --summary all
   ```
   *(Ignore stderr warnings.)*

### Documentation
Generated using [Zig's standard documentation format](https://ziglang.org/documentation/master/#Doc-Comments).

### Docker
Follow the [Docker Guide](How_TO_DOCKER_101.md) for containerized usage.

---

## Join Us!

Contribute to Zant on [GitHub](#). Let’s make NN deployment on microcontrollers efficient, accessible, and open!
