# TheTinyBook

## Project Overview

The aim of this project is to develop a Zig library designed to optimize the training and inference of ML models on embedded systems and edge devices. Embedded systems and edge devices typically have limited computational resources, power constraints, and often operate in real-time environments. This library will address these challenges by providing optimized algorithms, efficient memory management, and low-level hardware interfacing tailored to the needs of AI model training on these constrained devices.
Key Features

 **Lightweight and High-Performance**: The library leverages Zig's low-level programming capabilities to deliver high-performance solutions while maintaining a small footprint, crucial for resource-constrained environments.

 **Hardware Acceleration Support**: The library will include support for various hardware acceleration techniques, such as SIMD (Single Instruction, Multiple Data) and GPU offloading, where applicable, to maximize the efficiency of AI model training.

 **Efficient Memory Management**: Given the limited memory availability in embedded systems, the library will incorporate advanced memory management techniques, including memory pooling, static memory allocation, and efficient buffer management.

 **Optimized Algorithms**: The library will provide a set of optimized algorithms specifically designed for AI model training on embedded devices. This includes quantization techniques, model pruning, and lightweight neural network architectures that reduce the computational load.

 **Cross-Platform Support**: The library will be designed to be cross-platform, supporting a variety of embedded systems, including ARM Cortex-M, RISC-V, and other popular microcontroller architectures.

 **Ease of Integration**: The library will be modular and easy to integrate into existing projects, with clear documentation, examples, and APIs that make it straightforward for developers to incorporate AI training capabilities into their embedded or edge devices.

## Use Cases
 
 **Real-Time Applications**: Enabling real-time AI applications such as anomaly detection, predictive maintenance, and object recognition on edge devices.
 **IoT Devices**: Integrating AI capabilities into IoT devices that operate in constrained environments, enhancing their ability to learn and adapt in real-time.
 **Autonomous Systems**: Supporting the training of AI models in autonomous systems like drones, robots, and vehicles where edge processing is crucial.

## Getting Started

### Prerequisites
 Zig Compiler: Ensure you have the latest version of the Zig compiler installed. Follow the guide on the official [website](https://ziglang.org/learn/getting-started/).
 Zig knowledge: To better understand the library is necessary a solid knowledge of zig language. We reccomend you a hands-on learning approach, try to solve those excercises... [ziglings/exercises](https://codeberg.org/ziglings/exercises)

# Run
Go on the project folder and digit:
 ```
 zig build run
```

# Test
 Every time you create a test_file.zig, if not already present remember to add his path  into build.zig/test_list.
 To run tests run: 
 
 ```
 zig build
 
 zig build test_all --summary all 
 ```
 
 (don't worry about stderr)

# Doc
For the documentation has been used the [ZIG standard](https://ziglang.org/documentation/master/#Doc-Comments)
