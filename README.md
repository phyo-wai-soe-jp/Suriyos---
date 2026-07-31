# Suriyos

A 3D impact/drop physics simulator for macOS, built with [Dear ImGui](https://github.com/ocornut/imgui), GLFW, and [Bullet Physics](https://pybullet.org/wordpress/index.php/forum-2/). Supports English/Japanese UI and exports simulation results for analysis.

## Requirements

- macOS 11+ (Apple Silicon or Intel)
- Xcode Command Line Tools (`xcode-select --install`)
- [Homebrew](https://brew.sh)

## Setup

```sh
brew install glfw bullet
git clone --depth 1 https://github.com/ocornut/imgui.git third_party/imgui
```

## Build & run

```sh
make        # build the Suriyos binary
make run    # build and launch
make app    # bundle into Suriyos.app (with icon)
```

## Project layout

```
Suriyos/
├── src/main.cpp        # application source
├── Resources/           # Info.plist + app icon used by `make app`
└── Makefile
```
