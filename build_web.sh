#!/usr/bin/env bash
# Builds the WebAssembly/WebGL2 version of Suriyos via Emscripten.
set -euo pipefail
cd "$(dirname "$0")"

export EMSDK="$HOME/emsdk"
export EMSDK_NODE="$EMSDK/node/22.16.0_64bit/bin/node"
export EMSDK_PYTHON="$EMSDK/python/3.13.3_64bit/bin/python3"
export PATH="$EMSDK:$EMSDK/upstream/emscripten:$PATH"

BULLET_SRC="../bullet3/src"
IMGUI="third_party/imgui"
OUT="web-build"
mkdir -p "$OUT"

emcc -std=c++17 -O2 -DNDEBUG \
  -I"$IMGUI" -I"$IMGUI/backends" -I"$BULLET_SRC" \
  src/main.cpp \
  "$IMGUI/imgui.cpp" "$IMGUI/imgui_draw.cpp" "$IMGUI/imgui_tables.cpp" "$IMGUI/imgui_widgets.cpp" \
  "$IMGUI/backends/imgui_impl_glfw.cpp" "$IMGUI/backends/imgui_impl_opengl3.cpp" \
  "$BULLET_SRC/btBulletDynamicsAll.cpp" "$BULLET_SRC/btBulletCollisionAll.cpp" "$BULLET_SRC/btLinearMathAll.cpp" \
  -s USE_GLFW=3 \
  -s USE_WEBGL2=1 \
  -s FULL_ES3=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s NO_EXIT_RUNTIME=1 \
  -s WASM=1 \
  -s EXPORTED_RUNTIME_METHODS='["FS"]' \
  --preload-file web-assets/DejaVuSans.ttf@/fonts/DejaVuSans.ttf \
  --shell-file shell.html \
  -o "$OUT/index.html"

echo "Built $OUT/suriyos.html"
