# Suriyos — macOS build
#
# Dependencies (installed via Homebrew):
#   brew install glfw bullet
#
# Dear ImGui is not vendored in this repo. Clone it once into third_party/imgui:
#   git clone --depth 1 https://github.com/ocornut/imgui.git third_party/imgui
#
# Usage:
#   make          build the "Suriyos" binary
#   make run      build and launch it
#   make app      bundle into Suriyos.app (with icon)
#   make clean    remove build output

APP_NAME    := Suriyos
IMGUI_DIR   ?= third_party/imgui
BULLET_PREFIX := $(shell brew --prefix bullet)
GLFW_PREFIX   := $(shell brew --prefix glfw)

CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -DGL_SILENCE_DEPRECATION -DGLFW_INCLUDE_NONE \
            -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends \
            -I$(GLFW_PREFIX)/include \
            -I$(BULLET_PREFIX)/include/bullet

LDFLAGS  := -L$(GLFW_PREFIX)/lib -lglfw -framework OpenGL \
            -L$(BULLET_PREFIX)/lib -lBulletDynamics -lBulletCollision -lLinearMath

SRC := src/main.cpp \
       $(IMGUI_DIR)/imgui.cpp \
       $(IMGUI_DIR)/imgui_draw.cpp \
       $(IMGUI_DIR)/imgui_tables.cpp \
       $(IMGUI_DIR)/imgui_widgets.cpp \
       $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
       $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

.PHONY: all run app clean

all: $(APP_NAME)

$(APP_NAME): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(APP_NAME)

run: all
	./$(APP_NAME)

app: all
	rm -rf $(APP_NAME).app
	mkdir -p $(APP_NAME).app/Contents/MacOS
	mkdir -p $(APP_NAME).app/Contents/Resources
	cp $(APP_NAME) $(APP_NAME).app/Contents/MacOS/$(APP_NAME)
	cp Resources/Info.plist $(APP_NAME).app/Contents/Info.plist
	cp Resources/AppIcon.icns $(APP_NAME).app/Contents/Resources/AppIcon.icns
	@echo "Built $(APP_NAME).app"

clean:
	rm -f $(APP_NAME)
	rm -rf $(APP_NAME).app
