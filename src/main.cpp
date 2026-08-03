// Physics Experiment — 3D impact/drop simulator with Bullet Physics and ImGui.
//
// macOS build (recommended):
//   make        (uses Makefile in this directory)
//
// Manual macOS build:
//   clang++ -std=c++17 Untitled-y.cpp \
//     -I/Users/phyowaisoe/Physics\ Experiment/imgui \
//     -I/opt/homebrew/opt/glfw/include -I/opt/homebrew/include/bullet \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui.cpp \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui_draw.cpp \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui_tables.cpp \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui_widgets.cpp \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui_impl_glfw.cpp \
//     /Users/phyowaisoe/Physics\ Experiment/imgui/imgui_impl_opengl3.cpp \
//     -L/opt/homebrew/opt/glfw/lib -lglfw -framework OpenGL \
//     -L/opt/homebrew/lib -lBulletDynamics -lBulletCollision -lLinearMath \
//     -o Untitled-y

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#if defined(__EMSCRIPTEN__)
#include <GLES3/gl3.h>
#include <emscripten/html5.h>
#else
#include <OpenGL/gl3.h>
#endif
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <btBulletDynamicsCommon.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

constexpr float pi = 3.14159265358979323846f;

// ── Language / Localisation ───────────────────────────────────────────────────
enum class Lang { EN = 0, JA = 1 };
static Lang gLang  = Lang::JA;   // default: Japanese
static int  gTheme = 0;          // 0=Midnight  1=Solar  2=Mint  3=Bloom  4=Rose  (macOS system accents)

static inline const char* T(const char* en, const char* ja) {
    return (gLang == Lang::JA) ? ja : en;
}

struct Vec3 {
    float x, y, z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 v, float scale) { return {v.x * scale, v.y * scale, v.z * scale}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalize(Vec3 v) {
    float length = std::sqrt(dot(v, v));
    return length > 0.0f ? v * (1.0f / length) : Vec3{0.0f, 1.0f, 0.0f};
}

struct Mat4 {
    float m[16]{};
};

Mat4 identity() {
    Mat4 result{};
    result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;
    return result;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result.m[column * 4 + row] += a.m[k * 4 + row] * b.m[column * 4 + k];
            }
        }
    }
    return result;
}

Mat4 translation(Vec3 value) {
    Mat4 result = identity();
    result.m[12] = value.x;
    result.m[13] = value.y;
    result.m[14] = value.z;
    return result;
}

Mat4 perspective(float fovRadians, float aspect, float nearPlane, float farPlane) {
    float scale = 1.0f / std::tan(fovRadians * 0.5f);
    Mat4 result{};
    result.m[0] = scale / aspect;
    result.m[5] = scale;
    result.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    result.m[11] = -1.0f;
    result.m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    return result;
}

Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 forward = normalize(center - eye);
    Vec3 side = normalize(cross(forward, up));
    Vec3 cameraUp = cross(side, forward);
    Mat4 result = identity();
    result.m[0] = side.x;
    result.m[4] = side.y;
    result.m[8] = side.z;
    result.m[1] = cameraUp.x;
    result.m[5] = cameraUp.y;
    result.m[9] = cameraUp.z;
    result.m[2] = -forward.x;
    result.m[6] = -forward.y;
    result.m[10] = -forward.z;
    result.m[12] = -dot(side, eye);
    result.m[13] = -dot(cameraUp, eye);
    result.m[14] = dot(forward, eye);
    return result;
}

// ── Quaternion ────────────────────────────────────────────────────────────────
struct Quat { float w, x, y, z; };

Quat qNorm(Quat q) {
    float len = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (len < 1e-6f) return {1,0,0,0};
    return {q.w/len, q.x/len, q.y/len, q.z/len};
}

Quat qMul(Quat a, Quat b) {
    return { a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
             a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
             a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
             a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w };
}

Vec3 qRot(Quat q, Vec3 v) {
    Vec3 u{q.x, q.y, q.z};
    return u * (2.0f * dot(u, v))
         + v  * (q.w*q.w - dot(u, u))
         + cross(u, v) * (2.0f * q.w);
}

Mat4 quatToMat4(Quat q) {
    Mat4 m = identity();
    float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    m.m[0]=1-2*(yy+zz); m.m[1]=2*(xy+wz);   m.m[2]=2*(xz-wy);
    m.m[4]=2*(xy-wz);   m.m[5]=1-2*(xx+zz); m.m[6]=2*(yz+wx);
    m.m[8]=2*(xz+wy);   m.m[9]=2*(yz-wx);   m.m[10]=1-2*(xx+yy);
    return m;
}

Quat eulerToQuat(float pitchDeg, float yawDeg, float rollDeg) {
    float p = pitchDeg * pi / 180.0f * 0.5f;
    float y = yawDeg   * pi / 180.0f * 0.5f;
    float r = rollDeg  * pi / 180.0f * 0.5f;
    float cp=std::cos(p),sp=std::sin(p);
    float cy=std::cos(y),sy=std::sin(y);
    float cr=std::cos(r),sr=std::sin(r);
    return qNorm({cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                  cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy});
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
    float r, g, b, a;
};

struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei count = 0;
    GLenum mode = GL_TRIANGLES;
    bool indexed = false;
};

float yaw = 35.0f;
float pitch = 25.0f;
float cameraDistance = 55.0f;
bool autoRotate = false;
bool dragging = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;
int framebufferWidth = 900;
int framebufferHeight = 700;

// ── Material / model database ─────────────────────────────────────────────────
// Scientific note:
// These are representative engineering values, not exact constants. Real material
// properties depend on alloy/purity, heat treatment, temperature, strain rate,
// surface condition, and measurement method.
//
// The simulator separates measured inputs from model assumptions. Any value that
// cannot be derived from the chosen geometry/materials must be confirmed by the
// user before it is used as a scientific result.
//
// Bullet handles rigid-body motion. Extra deformation/fracture/crater logic below
// is an educational model unless calibrated against experiment.

#include "sim/materials.h"   // ObjectMaterial, GroundMaterial, kObjectMaterials, kGroundMaterials, kGravities, kAtmospheres

// ── Japanese display names for material / environment tables ──────────────────
static const char* kGroundMatNamesJA[] = {
    "鋼板", "コンクリート", "乾燥土", "木質床", "氷面", "砂地", "ゴムマット"
};
static const char* kObjMatNamesJA[] = {
    "鉄 (Fe-56)", "銅 (Cu-63)", "アルミ (Al-27)", "鉛 (Pb-208)",
    "チタン (Ti-48)", "木材（オーク）", "ゴム", "氷 (H₂O)"
};
static const char* kGravNamesJA[] = {
    "地球", "月", "火星", "木星", "海王星", "冥王星", "太陽", "無重力"
};
static const char* kAtmNamesJA[] = {
    "真空", "火星（希薄CO₂）", "地球（海面）", "金星（高密度CO₂）", "水中"
};
static const char* kColorNamesJA[] = {
    "グレー", "シアン", "オレンジ", "グリーン", "マゼンタ", "イエロー", "ホワイト"
};

int objectMaterialIdx = 0;   // Iron
int groundMaterialIdx = 0;   // Steel plate
int gravityEnvIdx     = 0;   // Earth
int atmosphereIdx     = 2;   // Earth air

// ── Object / terrain state ────────────────────────────────────────────────────
int gridSize = 500;
float gridSpacing = 1.0f;
float ballRadius    = 0.5f;
float ballDropHeight = 8.0f;   // metres from object bottom to terrain at (0,0)
Vec3 ballPosition{0.0f, ballDropHeight, 0.0f};
Vec3 ballVelocity{0.0f, 0.0f, 0.0f};
Quat objectOrientation{1.0f, 0.0f, 0.0f, 0.0f};
Vec3 angularVelocity{0.0f, 0.0f, 0.0f};

constexpr float gravity      = -9.8f;
constexpr float bounceFactor =  0.72f;
constexpr float friction     =  0.50f;

// ── Bullet Physics globals ────────────────────────────────────────────────────
static btDefaultCollisionConfiguration*      gBtConfig   = nullptr;
static btCollisionDispatcher*                gBtDispatch = nullptr;
static btDbvtBroadphase*                     gBtBP       = nullptr;
static btSequentialImpulseConstraintSolver*  gBtSolver   = nullptr;
static btDiscreteDynamicsWorld*              gBtWorld    = nullptr;
static btRigidBody*                          gBtObject   = nullptr;
static btDefaultMotionState*                 gBtObjMS    = nullptr;
static btCollisionShape*                     gBtObjShape = nullptr;
static btRigidBody*                          gBtGround   = nullptr;
static btDefaultMotionState*                 gBtGndMS    = nullptr;
static btCollisionShape*                     gBtGndShape = nullptr;
static btTriangleMesh*                       gBtMesh     = nullptr;

struct Color {
    float r, g, b;
    const char* name;
};

Color gridColors[] = {
    {0.30f, 0.30f, 0.35f, "Slate"},   {0.20f, 0.80f, 0.90f, "Cyan"},
    {0.90f, 0.40f, 0.20f, "Orange"},  {0.50f, 0.90f, 0.40f, "Green"},
    {0.85f, 0.30f, 0.70f, "Magenta"}, {0.95f, 0.85f, 0.30f, "Yellow"},
    {0.90f, 0.90f, 0.95f, "White"},
};
constexpr int numColors = sizeof(gridColors) / sizeof(gridColors[0]);
int currentColor = 0;

GLuint shaderProgram = 0;
GLint mvpLocation      = -1;
GLint modelLocation    = -1;
GLint litLocation      = -1;
GLint specularLocation = -1;
GLint eyeLocation      = -1;
GLint colorMultLocation= -1;
GLint fogStartLocation = -1;
GLint fogEndLocation   = -1;

// Terrain follows the ball so the ground extends infinitely in any direction.
float gTerrainCX = 0.0f;
float gTerrainCZ = 0.0f;
Mesh terrainMesh;
Mesh gridMesh;
Mesh axesMesh;
Mesh sphereMesh;
Mesh boxMesh;
Mesh shadowMesh;
Mesh fragmentMesh;   // shared small-sphere mesh reused for all shard visuals
Mesh gUnitSphereMesh;
Mesh gUnitBoxMesh;
Mesh gCylinderMesh;
Mesh gConeMesh;
Mesh gCapsuleMesh;
Mesh gCarMesh;

// ── Fracture / fragment state ─────────────────────────────────────────────────
bool  gObjectFragmented = false;
float gLastImpactVel    = 0.0f;    // recorded for HUD display
Vec3  gPrevVelocity{0,0,0};        // velocity one step before impact

struct Fragment {
    btRigidBody*          body;
    btCollisionShape*     shape;
    btDefaultMotionState* ms;
    float sx, sy, sz;   // non-uniform visual scale per shard
};
std::vector<Fragment> gFragments;

// ── Scene Objects (user-placed, individually controllable) ────────────────────
constexpr const char* kSceneShapeNames[] = {"Sphere","Box","Cylinder","Cone","Capsule","Car"};
constexpr int kNumSceneShapes = 6;

static const char* const* getSceneShapeNamesL() {
    static const char* en[] = {"Sphere","Box","Cylinder","Cone","Capsule","Car"};
    static const char* ja[] = {"球体","直方体","円柱","円錐","カプセル体","車"};
    return (gLang == Lang::JA) ? ja : en;
}

struct SceneObject {
    int   shapeType = 0;
    int   matIdx    = 0;
    float r         = 0.5f;
    Vec3  pos       = {0,5,0};
    Vec3  euler     = {0,0,0};   // pitch, yaw, roll degrees (UI-only; physics takes over after spawn)
    Quat  orient    = {1,0,0,0};
    Vec3  vel       = {0,0,0};
    bool  forceOn       = false;
    Vec3  forceDir      = {0,1,0};
    float forceMag      = 0.0f;   // Newtons, continuous
    float impulseStrength = 500.0f; // N·s, one-shot fire
    float sx=1.f, sy=1.f, sz=1.f;  // non-uniform scale (workshop)
    char  label[32] = {};
    btRigidBody*          body   = nullptr;
    btCollisionShape*     bshape = nullptr;
    btDefaultMotionState* bms    = nullptr;
};

std::vector<SceneObject> gSceneObjects;
int   gSelectedObjIdx = -1;
int   gLibShape       = 0;
int   gLibMat         = 0;
float gLibSize        = 0.5f;

// ── Object Library data ───────────────────────────────────────────────────────
struct LibVariant { const char* name; float sx,sy,sz; int matHint; };
struct LibEntry {
    const char* name; const char* category; int baseShape;
    const char* desc;
    LibVariant variants[4]; int nVariants;
    float r,g,b;
};
static const LibEntry kLibEntries[] = {
    {"Sphere",      "Primitives",0,"Uniform sphere — smooth rolling, Hertz contact",
     {{"Standard",1.f,1.f,1.f,0},{"Oblate",1.8f,0.4f,1.8f,0},{"Prolate",0.5f,2.0f,0.5f,0},{"Heavy",1.f,1.f,1.f,3}},4, 0.85f,0.85f,0.90f},
    {"Metal Ball",  "Primitives",0,"Dense solid — good for high-energy impact tests",
     {{"Iron",1.f,1.f,1.f,0},{"Copper",1.f,1.f,1.f,1},{"Lead",0.9f,0.9f,0.9f,3},{"Titanium",1.f,1.f,1.f,4}},4, 0.60f,0.61f,0.65f},
    {"Ice Sphere",  "Primitives",0,"Brittle — fractures at critical Hertz velocity",
     {{"Hailstone",0.8f,0.8f,0.8f,7},{"Ice Egg",0.8f,1.3f,0.8f,7},{"Ice Ball",1.f,1.f,1.f,7},{"Spike",0.4f,2.5f,0.4f,7}},4, 0.75f,0.90f,0.98f},
    {"Rubber Ball", "Primitives",0,"High restitution — bounces repeatedly",
     {{"Bounce",1.f,1.f,1.f,6},{"Flat",1.5f,0.5f,1.5f,6},{"Egg",0.8f,1.4f,0.8f,6},{"Tiny",0.4f,0.4f,0.4f,6}},4, 0.15f,0.15f,0.15f},
    {"Cube",        "Boxes",    1,"Box — tumbles on corners, interesting torque dynamics",
     {{"Cube",1.f,1.f,1.f,0},{"Brick",2.f,1.f,1.f,5},{"Plank",3.f,0.2f,1.5f,5},{"Beam",0.4f,0.4f,4.f,5}},4, 0.60f,0.61f,0.65f},
    {"Wood Block",  "Boxes",    1,"Low density, moderate bounce, high friction",
     {{"Block",1.f,1.f,1.f,5},{"Log",0.5f,0.5f,3.f,5},{"Board",2.5f,0.15f,1.f,5},{"Post",0.3f,3.f,0.3f,5}},4, 0.55f,0.37f,0.18f},
    {"Ice Block",   "Boxes",    1,"Brittle box — shatters on hard surfaces",
     {{"Cube",1.f,1.f,1.f,7},{"Slab",2.f,0.4f,2.f,7},{"Wall",3.f,2.f,0.3f,7},{"Tower",0.8f,3.f,0.8f,7}},4, 0.75f,0.90f,0.98f},
    {"Cylinder",    "Cylinders",2,"Rolls steadily — good for ramp/slope experiments",
     {{"Standard",1.f,1.f,1.f,0},{"Disk",2.f,0.3f,2.f,0},{"Rod",0.3f,4.f,0.3f,0},{"Barrel",1.3f,0.9f,1.3f,5}},4, 0.72f,0.45f,0.20f},
    {"Wheel",       "Cylinders",2,"Wide disk — rolls and can steer on slopes",
     {{"Thin",1.5f,0.25f,1.5f,2},{"Fat",1.f,0.5f,1.f,6},{"Heavy",1.5f,0.4f,1.5f,0},{"Giant",2.f,0.5f,2.f,2}},4, 0.52f,0.54f,0.58f},
    {"Cone",        "Cones",    3,"Unstable equilibrium — tips over predictably",
     {{"Standard",1.f,1.f,1.f,0},{"Flat",2.5f,0.6f,2.5f,0},{"Spike",0.4f,3.f,0.4f,0},{"Wide",1.8f,0.8f,1.8f,5}},4, 0.85f,0.85f,0.50f},
    {"Capsule",     "Capsules", 4,"Smooth ends — stable rolling, minimal snagging",
     {{"Standard",1.f,1.f,1.f,2},{"Long",0.6f,2.5f,0.6f,2},{"Fat",1.5f,1.f,1.5f,4},{"Pill",0.8f,1.5f,0.8f,6}},4, 0.72f,0.45f,0.20f},
    // ── Cars ──────────────────────────────────────────────────────────────────
    {"Land Cruiser","Cars",    5,"Full-size SUV — heavy frame, high ground clearance",
     {{"Standard",2.0f,1.9f,5.0f,2},{"Off-Road",2.0f,2.1f,5.1f,0},{"Sport",2.0f,1.8f,4.9f,2},{"Armored",2.1f,2.0f,5.2f,0}},4, 0.82f,0.82f,0.84f},
    {"Van",         "Cars",    5,"High-roof van — large cargo volume, boxy profile",
     {{"Passenger",2.1f,2.3f,5.4f,2},{"Cargo",2.1f,2.2f,5.4f,0},{"Mini",1.9f,2.0f,4.6f,2},{"Long",2.1f,2.4f,6.2f,0}},4, 0.90f,0.90f,0.92f},
    {"Sports Car",  "Cars",    5,"Low-slung coupe — wide stance, aerodynamic body",
     {{"Coupe",2.0f,1.1f,4.4f,2},{"Roadster",2.0f,1.0f,4.3f,2},{"GT",2.05f,1.15f,4.6f,4},{"Track",2.0f,0.95f,4.5f,2}},4, 0.85f,0.12f,0.08f},
    {"Pickup Truck","Cars",    5,"Body-on-frame truck — open cargo bed, rugged chassis",
     {{"Standard",1.9f,1.8f,5.3f,0},{"Lifted",1.9f,2.0f,5.3f,0},{"Short Bed",1.9f,1.8f,4.8f,0},{"Heavy",2.0f,1.9f,5.8f,0}},4, 0.35f,0.35f,0.38f},
    {"Sedan",       "Cars",    5,"Classic 3-box saloon — balanced proportions",
     {{"Compact",1.8f,1.5f,4.6f,2},{"Mid-size",1.85f,1.5f,4.9f,2},{"Full-size",1.9f,1.55f,5.2f,2},{"Sport",1.85f,1.42f,4.8f,2}},4, 0.20f,0.35f,0.75f},
    {"Compact SUV", "Cars",    5,"Crossover SUV — raised ride height, unibody frame",
     {{"Standard",1.9f,1.7f,4.6f,2},{"Tall",1.9f,1.8f,4.6f,2},{"Sport",1.9f,1.65f,4.6f,2},{"AWD",1.9f,1.75f,4.7f,0}},4, 0.15f,0.25f,0.60f},
    {"City Bus",    "Cars",    5,"Transit bus — rigid body, high passenger capacity",
     {{"Standard",2.6f,3.1f,12.0f,5},{"Coach",2.6f,3.2f,13.7f,5},{"Mini Bus",2.3f,2.8f,7.0f,2},{"Double Decker",2.5f,5.5f,11.0f,5}},4, 0.90f,0.75f,0.10f},
    {"Semi Cab",    "Cars",    5,"Tractor cab unit — high hood, sleeper behind cab",
     {{"Day Cab",2.5f,3.8f,6.5f,0},{"Sleeper",2.5f,3.9f,8.0f,0},{"Flat Nose",2.6f,3.6f,6.0f,0},{"Heavy",2.6f,4.0f,7.5f,0}},4, 0.65f,0.65f,0.70f},
    {"Hatchback",   "Cars",    5,"Compact 5-door — tall cabin relative to length",
     {{"3-Door",1.75f,1.52f,3.9f,2},{"5-Door",1.75f,1.54f,4.0f,2},{"Hot Hatch",1.78f,1.45f,4.0f,2},{"Micro",1.6f,1.5f,3.5f,2}},4, 0.20f,0.65f,0.25f},
    {"Formula Car", "Cars",    5,"Open-wheel racer — ultra-low, wide front wing",
     {{"F1 Style",2.0f,0.95f,5.3f,2},{"Indy",2.0f,1.0f,5.5f,2},{"Kart",1.4f,0.6f,2.5f,4},{"GT Race",2.1f,1.05f,4.9f,2}},4, 1.0f,0.20f,0.05f},
};
static const int kNumLibEntries = (int)(sizeof(kLibEntries)/sizeof(kLibEntries[0]));
static const char* kLibCategories[] = {"All","Primitives","Boxes","Cylinders","Cones","Capsules","Cars"};
static const char* kLibCategoriesJA[] = {"すべて","基本形","直方体","円柱","円錐","カプセル","車"};
static const int   kNumLibCats = 7;
static const char* kLibEntryNamesJA[] = {
    "球体","金属球","氷球","ゴム球",
    "立方体","木材ブロック","氷ブロック",
    "円柱体","車輪","円錐体","カプセル体",
    "ランドクルーザー","バン","スポーツカー","ピックアップトラック",
    "セダン","コンパクトSUV","路線バス","セミキャブ",
    "ハッチバック","フォーミュラカー"
};
static const char* kLibVariantNamesJA[][4] = {
    {"標準","扁平","縦長","重量型"},
    {"鉄","銅","鉛","チタン"},
    {"ひょう","卵形氷","氷球","棘形"},
    {"弾性","扁平","卵形","小型"},
    {"立方体","レンガ","板材","梁"},
    {"ブロック","丸太","板","柱"},
    {"立方体","板氷","壁氷","塔氷"},
    {"標準","円盤","棒","樽"},
    {"細型","太型","重量型","大型"},
    {"標準","扁平","尖型","広型"},
    {"標準","長型","太型","錠剤型"},
    {"標準","オフロード","スポーツ","装甲"},
    {"乗用","貨物","ミニ","ロング"},
    {"クーペ","ロードスター","GT","サーキット"},
    {"標準","リフトアップ","ショートベッド","重量型"},
    {"コンパクト","ミッドサイズ","フルサイズ","スポーツ"},
    {"標準","ハイライド","スポーツ","AWD"},
    {"標準","コーチ","ミニバス","2階建て"},
    {"デイキャブ","スリーパー","フラットノーズ","重量型"},
    {"3ドア","5ドア","ホットハッチ","マイクロ"},
    {"F1スタイル","インディ","カート","GTレース"},
};

// Library UI state
int   gLibCatFilter  = 0;
int   gLibSelected   = -1;
int   gLibVariantSel = 0;
float gLibSx=1.f,gLibSy=1.f,gLibSz=1.f;

// Payload transferred during a library-card drag-drop operation
struct LibDragPayload { int entryIdx; int variantIdx; };

// ── Scene constraints (workshop — Connect tab) ────────────────────────────────
constexpr const char* kConstraintNames[] = {"Point-to-Point","Hinge","Spring","Fixed"};
constexpr int kNumConstraintTypes = 4;

static const char* const* getConstraintNamesL() {
    static const char* en[] = {"Point-to-Point","Hinge","Spring","Fixed"};
    static const char* ja[] = {"点対点拘束","ヒンジ拘束","バネ拘束","固定拘束"};
    return (gLang == Lang::JA) ? ja : en;
}
struct SceneConstraint {
    int   typeIdx=0, objA=-1, objB=-1;
    Vec3  pivotA{0,0,0}, pivotB{0,0,0};
    Vec3  hingeAxis{0,1,0};
    float limitLow=-90.f, limitHigh=90.f;
    float springK=500.f, springD=20.f;
    char  label[48]={};
    btTypedConstraint* bt=nullptr;
};
std::vector<SceneConstraint> gSceneConstraints;

// Workshop UI state
int   gWsConnA=-1, gWsConnB=-1, gWsConType=0;
Vec3  gWsPivA{0,0,0}, gWsPivB{0,0,0}, gWsHingeAx{0,1,0};
float gWsLimitLow=-90.f,gWsLimitHigh=90.f,gWsSpringK=500.f,gWsSpringD=20.f;
float gWsSx=1.f,gWsSy=1.f,gWsSz=1.f;
int   gWsShape=0, gWsMat=0;
float gWsRadius=0.5f;

// ── Simulation pause/step ──────────────────────────────────────────────────────
bool gSimPaused = false;
bool gSimStep   = false;   // advance exactly one physics tick then re-pause

// ── Transform gizmo ──────────────────────────────────────────────────────────
enum class GizmoMode { Translate = 0, Rotate = 1, Scale = 2 };
GizmoMode gGizmoMode = GizmoMode::Translate;

struct GizmoDrag {
    bool   active   = false;
    int    axis     = -1;   // 0=X 1=Y 2=Z 3=center
    double startMX  = 0, startMY = 0;
    Vec3   startPos = {};
    Vec3   startEuler = {};
    float  startSx=1.f, startSy=1.f, startSz=1.f;
};
GizmoDrag gGizmoDrag;
Mesh      gGizmoLineMesh;

// Single click → Translate, double click → Rotate
static double gLastClickTime  = -1.0;
static int    gLastClickedObj = -1;
constexpr double kDoubleClickSec = 0.35;

// ── Time-series recording ──────────────────────────────────────────────────────
struct TimeSeriesFrame {
    float t;                // simulation time (s)
    Vec3  pos, vel;         // ball position + velocity
    float speed;            // |vel| m/s
    float ke;               // kinetic energy J
    float altitude;         // height above ground
};
static constexpr int kMaxFrames = 8000;   // ~133 s at 60 Hz
std::vector<TimeSeriesFrame> gTimeSeries;
bool  gRecording  = false;
float gSimTime    = 0.0f;   // running simulation clock (reset with ball)
// Impact snapshot (filled on first detected contact during a recording)
struct ImpactSnapshot {
    bool   valid       = false;
    float  time        = 0.0f;
    float  speed       = 0.0f;   // m/s
    float  impulse     = 0.0f;   // N·s  (filled from contact manifold)
    int    craterCount = 0;
    Vec3   pos         = {};
};
ImpactSnapshot gImpactSnap;

// ── Impact flash ──────────────────────────────────────────────────────────────
Vec3  gFlashPos{0,0,0};
float gFlashTimer   = 0.0f;
float gFlashImpulse = 0.0f;  // last recorded contact impulse (N·s)
Mesh  gFlashMesh;

// ── Velocity trail ────────────────────────────────────────────────────────────
struct TrailPt { Vec3 pos; };
std::deque<TrailPt> gTrail;
constexpr int TRAIL_MAX = 70;
Mesh  gTrailDotMesh;
bool  gShowTrail = true;

// ── Dynamic terrain craters (soil / sand deformation) ────────────────────────
struct Crater { float cx, cz, depth, radius; };
std::vector<Crater> gCraters;
bool  gTerrainDirty = false;

// ── Live aerodynamic readouts for HUD ─────────────────────────────────────────
float gAeroDragN   = 0.0f;
float gAeroMagnusN = 0.0f;

int groundType    = 0;  // 0=Flat  1=Bumpy Math  2=Rolling Hills  3=Bowl
int objectShape   = 0;  // 0=Sphere  1=Box
int objectSizeIdx = 1;  // 0=S(0.25)  1=M(0.5)  2=L(0.8)  3=XL(1.2)
constexpr float kObjectSizes[] = {0.25f, 0.5f, 0.8f, 1.2f};


// ── Scientific model controls ────────────────────────────────────────────────
// RigidBody: Bullet rigid-body result only. No exact deformation is claimed.
// IdealElastic: computes ideal spring-like compression estimates for HUD/report.
// ElasticPlastic: reserved for a future yield/plasticity model; not enabled yet.
enum class PhysicsModel { RigidBody = 0, IdealElastic = 1, ElasticPlastic = 2 };
PhysicsModel gPhysicsModel = PhysicsModel::RigidBody;

// Important: contact duration is not known from the stated problem alone.
// If this stays <= 0, code must not convert impulse to force/pressure for claims.
float gContactDurationEstimate = 0.0f; // seconds; user-confirmed/calibrated only

struct AssumptionReport {
    bool contactDurationConfirmed = false;
    bool targetSupportConfirmed   = false;
    bool materialSampleConfirmed  = false;
};
AssumptionReport gAssumptions{};

// For the user's 1 m cube case, objectShape=Box and ballRadius=0.5f because the
// box mesh and Bullet box use half-extent r, so side length = 2r = 1 m.
void configureOneMeterIronCubeDropPreset() {
    objectMaterialIdx = 0;  // Iron representative value
    gravityEnvIdx     = 0;  // Earth
    atmosphereIdx     = 2;  // Earth air; change to 0 for vacuum after confirmation
    objectShape       = 1;  // Box/cube
    ballRadius        = 0.5f; // half-side; cube side = 1 m
    ballDropHeight    = 10.0f; // bottom face 10 m above terrain/top surface
    groundType        = 0;  // flat surface; not a 3 m target cube yet
    groundMaterialIdx = 0;  // steel plate placeholder, not confirmed as 3 m iron cube
}

// Global window handle and frame timer (needed for Emscripten main-loop callback).
GLFWwindow* gWindow = nullptr;
double previousTime = 0.0;

float groundHeight(float x, float z) {
    float h;
    switch (groundType) {
        case 0:  h = 0.0f; break;
        case 2:  h = 1.2f * std::sin(x * 0.18f) * std::cos(z * 0.15f)
                     + 0.5f * std::sin((x + z) * 0.28f); break;
        case 3:  h = (x * x + z * z) * 0.003f - 2.0f; break;
        default: h = 0.28f * std::sin(x * 0.72f) * std::cos(z * 0.58f)
                     + 0.12f * std::sin((x + z) * 1.35f); break;
    }
    // Subtract dynamic crater deformations (soil/sand impacts)
    for (const auto& c : gCraters) {
        float d = std::sqrt((x - c.cx)*(x - c.cx) + (z - c.cz)*(z - c.cz));
        if (d < c.radius) {
            float t = 1.0f - d / c.radius;
            h -= c.depth * t * t;   // smooth paraboloid crater
        }
    }
    return h;
}

Vec3 groundNormal(float x, float z) {
    constexpr float sampleDistance = 0.02f;
    float slopeX = (groundHeight(x + sampleDistance, z) - groundHeight(x - sampleDistance, z))
                 / (2.0f * sampleDistance);
    float slopeZ = (groundHeight(x, z + sampleDistance) - groundHeight(x, z - sampleDistance))
                 / (2.0f * sampleDistance);
    return normalize({-slopeX, 1.0f, -slopeZ});
}

void configureMesh(Mesh& mesh) {
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, r)));
    glBindVertexArray(0);
}

void uploadVertices(Mesh& mesh, const std::vector<Vertex>& vertices, GLenum usage = GL_STATIC_DRAW) {
    if (mesh.vao == 0) configureMesh(mesh);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), usage);
    mesh.count = static_cast<GLsizei>(vertices.size());
    mesh.indexed = false;
}

void uploadIndexedMesh(Mesh& mesh, const std::vector<Vertex>& vertices,
                       const std::vector<unsigned int>& indices) {
    if (mesh.vao == 0) configureMesh(mesh);
    if (mesh.ebo == 0) glGenBuffers(1, &mesh.ebo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(),
                 GL_STATIC_DRAW);
    glBindVertexArray(0);
    mesh.count = static_cast<GLsizei>(indices.size());
    mesh.indexed = true;
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compilation failed:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

void createShaderProgram() {
    // Desktop OpenGL 4.1 core shaders (GLSL ES 3.00 / WebGL2 on the web build).
#if defined(__EMSCRIPTEN__)
    const char* glslVersionVert = "#version 300 es\n";
    const char* glslVersionFrag = "#version 300 es\nprecision mediump float;\n";
#else
    const char* glslVersionVert = "#version 410 core\n";
    const char* glslVersionFrag = "#version 410 core\n";
#endif

    std::string vertexShaderBody = R"(
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec4 aColor;
        uniform mat4 uMvp;
        uniform mat4 uModel;
        out vec3 vNormal;
        out vec3 vWorldPos;
        out vec4 vColor;
        void main() {
            vec4 worldPos = uModel * vec4(aPosition, 1.0);
            gl_Position = uMvp * vec4(aPosition, 1.0);
            vNormal   = mat3(uModel) * aNormal;
            vWorldPos = worldPos.xyz;
            vColor    = aColor;
        }
    )";

    std::string fragmentShaderBody = R"(
        in vec3 vNormal;
        in vec3 vWorldPos;
        in vec4 vColor;
        uniform bool uLit;
        uniform bool uSpecular;
        uniform vec3 uEye;
        uniform vec3 uColorMult;
        uniform float uFogStart;
        uniform float uFogEnd;
        out vec4 fragmentColor;
        void main() {
            vec3 color = vColor.rgb * uColorMult;
            if (uLit) {
                vec3 n        = normalize(vNormal);
                vec3 lightDir = normalize(vec3(0.35, 1.0, 0.55));
                float diff    = 0.34 + 0.78 * max(dot(n, lightDir), 0.0);
                color *= diff;
                if (uSpecular) {
                    vec3 viewDir = normalize(uEye - vWorldPos);
                    vec3 halfVec = normalize(lightDir + viewDir);
                    float spec   = pow(max(dot(n, halfVec), 0.0), 32.0);
                    color += vec3(0.5) * spec;
                }
            }
            float fogDist   = length(uEye - vWorldPos);
            float fogFactor = clamp((fogDist - uFogStart) / (uFogEnd - uFogStart), 0.0, 1.0);
            color = mix(color, vec3(0.07, 0.07, 0.10), fogFactor);
            fragmentColor = vec4(color, vColor.a);
        }
    )";

    std::string vertexShaderSource = std::string(glslVersionVert) + vertexShaderBody;
    std::string fragmentShaderSource = std::string(glslVersionFrag) + fragmentShaderBody;
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource.c_str());
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource.c_str());
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked = GL_FALSE;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        glGetProgramInfoLog(shaderProgram, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader link failed:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }
    mvpLocation      = glGetUniformLocation(shaderProgram, "uMvp");
    modelLocation    = glGetUniformLocation(shaderProgram, "uModel");
    litLocation      = glGetUniformLocation(shaderProgram, "uLit");
    specularLocation = glGetUniformLocation(shaderProgram, "uSpecular");
    eyeLocation      = glGetUniformLocation(shaderProgram, "uEye");
    colorMultLocation= glGetUniformLocation(shaderProgram, "uColorMult");
    fogStartLocation = glGetUniformLocation(shaderProgram, "uFogStart");
    fogEndLocation   = glGetUniformLocation(shaderProgram, "uFogEnd");
    glUseProgram(shaderProgram);
    glUniform3f(colorMultLocation, 1.0f, 1.0f, 1.0f);
    glUniform1f(fogStartLocation, 100.0f);
    glUniform1f(fogEndLocation,   400.0f);
}

void createTerrainMesh() {
    constexpr int steps = 128;
    const float extent = gridSize * gridSpacing;
    float cx = gTerrainCX, cz = gTerrainCZ;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (int zIndex = 0; zIndex <= steps; ++zIndex) {
        for (int xIndex = 0; xIndex <= steps; ++xIndex) {
            float lx = -extent + 2.0f * extent * xIndex / steps;
            float lz = -extent + 2.0f * extent * zIndex / steps;
            float wx = cx + lx, wz = cz + lz;
            float y = groundHeight(wx, wz);
            float tint = 0.14f + y * 0.08f;
            vertices.push_back({{lx, y - 0.01f, lz}, groundNormal(wx, wz),
                                tint, tint + 0.01f, tint + 0.045f, 1.0f});
        }
    }
    for (int z = 0; z < steps; ++z) {
        for (int x = 0; x < steps; ++x) {
            unsigned int a = z * (steps + 1) + x;
            unsigned int b = a + 1;
            unsigned int c = a + steps + 1;
            unsigned int d = c + 1;
            indices.insert(indices.end(), {a, c, b, b, c, d});
        }
    }
    uploadIndexedMesh(terrainMesh, vertices, indices);
}

void createGridMesh() {
    constexpr int lineSteps = 64;
    const float extent = gridSize * gridSpacing;
    float cx = gTerrainCX, cz = gTerrainCZ;
    std::vector<Vertex> vertices;
    const Color& color = gridColors[currentColor];
    auto addVertex = [&](float lx, float lz) {
        float wx = cx + lx, wz = cz + lz;
        vertices.push_back({{lx, groundHeight(wx, wz) + 0.012f, lz}, {0.0f, 1.0f, 0.0f},
                            color.r, color.g, color.b, 0.80f});
    };
    for (int i = -gridSize; i <= gridSize; ++i) {
        float p = i * gridSpacing;
        for (int step = 0; step < lineSteps; ++step) {
            float a = -extent + 2.0f * extent * step / lineSteps;
            float b = -extent + 2.0f * extent * (step + 1) / lineSteps;
            addVertex(p, a);
            addVertex(p, b);
            addVertex(a, p);
            addVertex(b, p);
        }
    }
    gridMesh.mode = GL_LINES;
    uploadVertices(gridMesh, vertices);
}

void createAxesMesh() {
    const float length = gridSize * gridSpacing + 2.0f;
    std::vector<Vertex> vertices = {
        {{-length, 0.0f, 0.0f}, {}, 1.0f, 0.25f, 0.25f, 1.0f},
        {{ length, 0.0f, 0.0f}, {}, 1.0f, 0.25f, 0.25f, 1.0f},
        {{0.0f, -length, 0.0f}, {}, 0.25f, 1.0f, 0.25f, 1.0f},
        {{0.0f,  length, 0.0f}, {}, 0.25f, 1.0f, 0.25f, 1.0f},
        {{0.0f, 0.0f, -length}, {}, 0.35f, 0.55f, 1.0f, 1.0f},
        {{0.0f, 0.0f,  length}, {}, 0.35f, 0.55f, 1.0f, 1.0f},
    };
    axesMesh.mode = GL_LINES;
    uploadVertices(axesMesh, vertices);
}

void createSphereMesh() {
    constexpr int latitudeSteps = 48;
    constexpr int longitudeSteps = 64;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    for (int latitude = 0; latitude <= latitudeSteps; ++latitude) {
        float polar = pi * latitude / latitudeSteps;
        for (int longitude = 0; longitude <= longitudeSteps; ++longitude) {
            float azimuth = 2.0f * pi * longitude / longitudeSteps;
            Vec3 normal{std::sin(polar) * std::cos(azimuth), std::cos(polar),
                        std::sin(polar) * std::sin(azimuth)};
            vertices.push_back({normal * ballRadius, normal, kObjectMaterials[objectMaterialIdx].r,
                kObjectMaterials[objectMaterialIdx].g,
                kObjectMaterials[objectMaterialIdx].b, 1.0f});
        }
    }
    for (int latitude = 0; latitude < latitudeSteps; ++latitude) {
        for (int longitude = 0; longitude < longitudeSteps; ++longitude) {
            unsigned int a = latitude * (longitudeSteps + 1) + longitude;
            unsigned int b = a + longitudeSteps + 1;
            indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    uploadIndexedMesh(sphereMesh, vertices, indices);
}

void createBoxMesh() {
    float r = ballRadius;
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;

    auto addFace = [&](Vec3 n, Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
        unsigned int base = static_cast<unsigned int>(verts.size());
        for (const Vec3& p : {a, b, c, d})
            verts.push_back({p, n, kObjectMaterials[objectMaterialIdx].r,
                kObjectMaterials[objectMaterialIdx].g,
                kObjectMaterials[objectMaterialIdx].b, 1.0f});
        idx.insert(idx.end(), {base, base+2, base+1, base+1, base+2, base+3});
    };

    addFace({ 1,0,0}, {r,-r,-r}, {r,-r, r}, {r, r,-r}, {r, r, r});
    addFace({-1,0,0}, {-r,-r, r}, {-r,-r,-r}, {-r, r, r}, {-r, r,-r});
    addFace({0, 1,0}, {-r, r, r}, { r, r, r}, {-r, r,-r}, { r, r,-r});
    addFace({0,-1,0}, {-r,-r,-r}, { r,-r,-r}, {-r,-r, r}, { r,-r, r});
    addFace({0,0, 1}, { r,-r, r}, {-r,-r, r}, { r, r, r}, {-r, r, r});
    addFace({0,0,-1}, {-r,-r,-r}, { r,-r,-r}, {-r, r,-r}, { r, r,-r});

    uploadIndexedMesh(boxMesh, verts, idx);
}

// Irregular angular shard — faceted, not a smooth sphere.
// Non-uniform per-fragment scale in render() makes each shard look unique.
void createFragmentMesh() {
    constexpr int lat = 7, lon = 9;
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;
    // Apply noise to radii to make it angular
    for (int la = 0; la <= lat; ++la) {
        float polar = pi * la / lat;
        for (int lo = 0; lo <= lon; ++lo) {
            float az = 2.0f * pi * lo / lon;
            // Noise that makes jagged edges
            float noise = 0.72f + 0.28f * std::sin(la * 2.1f + lo * 1.7f);
            Vec3 unit{std::sin(polar)*std::cos(az), std::cos(polar), std::sin(polar)*std::sin(az)};
            Vec3 pos = unit * noise;
            // Approximate normal by using unit direction (close enough for angular shard)
            verts.push_back({pos, unit,
                kObjectMaterials[objectMaterialIdx].r,
                kObjectMaterials[objectMaterialIdx].g,
                kObjectMaterials[objectMaterialIdx].b, 1.0f});
        }
    }
    for (int la = 0; la < lat; ++la)
        for (int lo = 0; lo < lon; ++lo) {
            unsigned int a = la*(lon+1)+lo, b = a+lon+1;
            idx.insert(idx.end(), {a,b,a+1, a+1,b,b+1});
        }
    uploadIndexedMesh(fragmentMesh, verts, idx);
}

// Bright yellow-white flash sphere rendered at impact point
void createFlashMesh() {
    constexpr int lat = 8, lon = 12;
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;
    for (int la = 0; la <= lat; ++la) {
        float polar = pi * la / lat;
        for (int lo = 0; lo <= lon; ++lo) {
            float az = 2.0f * pi * lo / lon;
            Vec3 n{std::sin(polar)*std::cos(az), std::cos(polar), std::sin(polar)*std::sin(az)};
            verts.push_back({n, n, 1.0f, 0.92f, 0.35f, 1.0f}); // hot yellow
        }
    }
    for (int la = 0; la < lat; ++la)
        for (int lo = 0; lo < lon; ++lo) {
            unsigned int a = la*(lon+1)+lo, b = a+lon+1;
            idx.insert(idx.end(), {a,b,a+1, a+1,b,b+1});
        }
    uploadIndexedMesh(gFlashMesh, verts, idx);
}

// Tiny white dot for velocity trail
void createTrailDotMesh() {
    constexpr int lat = 4, lon = 6;
    std::vector<Vertex> verts;
    std::vector<unsigned int> idx;
    for (int la = 0; la <= lat; ++la) {
        float polar = pi * la / lat;
        for (int lo = 0; lo <= lon; ++lo) {
            float az = 2.0f * pi * lo / lon;
            Vec3 n{std::sin(polar)*std::cos(az), std::cos(polar), std::sin(polar)*std::sin(az)};
            verts.push_back({n, n, 0.85f, 0.85f, 1.0f, 1.0f}); // pale blue-white
        }
    }
    for (int la = 0; la < lat; ++la)
        for (int lo = 0; lo < lon; ++lo) {
            unsigned int a = la*(lon+1)+lo, b = a+lon+1;
            idx.insert(idx.end(), {a,b,a+1, a+1,b,b+1});
        }
    uploadIndexedMesh(gTrailDotMesh, verts, idx);
}

// ── Unit meshes for scene objects (all radius/half-extent = 1; scaled in model matrix) ──

void createUnitSphereMesh() {
    constexpr int lat = 24, lon = 32;
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    for (int la=0;la<=lat;++la) { float polar=pi*la/lat;
        for (int lo=0;lo<=lon;++lo) { float az=2.0f*pi*lo/lon;
            Vec3 n{std::sin(polar)*std::cos(az),std::cos(polar),std::sin(polar)*std::sin(az)};
            verts.push_back({n,n,1,1,1,1}); } }
    for (int la=0;la<lat;++la) for (int lo=0;lo<lon;++lo) {
        unsigned a=la*(lon+1)+lo, b=a+lon+1;
        idx.insert(idx.end(),{a,b,a+1,a+1,b,b+1}); }
    uploadIndexedMesh(gUnitSphereMesh,verts,idx);
}

void createUnitBoxMesh() {
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    auto face=[&](Vec3 n,Vec3 a,Vec3 b,Vec3 c,Vec3 d){
        unsigned base=static_cast<unsigned>(verts.size());
        for (const Vec3& p:{a,b,c,d}) verts.push_back({p,n,1,1,1,1});
        idx.insert(idx.end(),{base,base+2,base+1,base+1,base+2,base+3});};
    face({ 1,0,0},{1,-1,-1},{1,-1,1},{1,1,-1},{1,1,1});
    face({-1,0,0},{-1,-1,1},{-1,-1,-1},{-1,1,1},{-1,1,-1});
    face({0,1,0},{-1,1,1},{1,1,1},{-1,1,-1},{1,1,-1});
    face({0,-1,0},{-1,-1,-1},{1,-1,-1},{-1,-1,1},{1,-1,1});
    face({0,0,1},{1,-1,1},{-1,-1,1},{1,1,1},{-1,1,1});
    face({0,0,-1},{-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1});
    uploadIndexedMesh(gUnitBoxMesh,verts,idx);
}

void createCylinderMesh() {
    constexpr int seg=32;
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    for (int i=0;i<=seg;++i) { float a=2.0f*pi*i/seg;
        Vec3 n{std::cos(a),0,std::sin(a)};
        verts.push_back({{n.x,-1,n.z},n,1,1,1,1});
        verts.push_back({{n.x, 1,n.z},n,1,1,1,1}); }
    for (int i=0;i<seg;++i){unsigned b=i*2;idx.insert(idx.end(),{b,b+2,b+1,b+1,b+2,b+3});}
    unsigned cB=static_cast<unsigned>(verts.size());
    verts.push_back({{0,-1,0},{0,-1,0},1,1,1,1});
    for (int i=0;i<=seg;++i){float a=2.0f*pi*i/seg;verts.push_back({{std::cos(a),-1,std::sin(a)},{0,-1,0},1,1,1,1});}
    for (int i=0;i<seg;++i) idx.insert(idx.end(),{cB,cB+i+2,cB+i+1});
    unsigned cT=static_cast<unsigned>(verts.size());
    verts.push_back({{0,1,0},{0,1,0},1,1,1,1});
    for (int i=0;i<=seg;++i){float a=2.0f*pi*i/seg;verts.push_back({{std::cos(a),1,std::sin(a)},{0,1,0},1,1,1,1});}
    for (int i=0;i<seg;++i) idx.insert(idx.end(),{cT,cT+i+1,cT+i+2});
    uploadIndexedMesh(gCylinderMesh,verts,idx);
}

void createConeMesh() {
    constexpr int seg=32;
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    for (int i=0;i<=seg;++i){float a=2.0f*pi*i/seg;
        Vec3 sideN=normalize({std::cos(a),0.5f,std::sin(a)});
        verts.push_back({{std::cos(a),-1,std::sin(a)},sideN,1,1,1,1});
        verts.push_back({{0,1,0},sideN,1,1,1,1});}
    for (int i=0;i<seg;++i){unsigned b=i*2;idx.insert(idx.end(),{b,b+2,b+1});}
    unsigned cB=static_cast<unsigned>(verts.size());
    verts.push_back({{0,-1,0},{0,-1,0},1,1,1,1});
    for (int i=0;i<=seg;++i){float a=2.0f*pi*i/seg;verts.push_back({{std::cos(a),-1,std::sin(a)},{0,-1,0},1,1,1,1});}
    for (int i=0;i<seg;++i) idx.insert(idx.end(),{cB,cB+i+2,cB+i+1});
    uploadIndexedMesh(gConeMesh,verts,idx);
}

void createCapsuleMesh() {
    // Radius=1, cylindrical half-height=0.5 → total height=3 (matches btCapsuleShape(r, r) scaled by r)
    constexpr int lat=16, lon=24;
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    float H=0.5f;
    // top hemisphere rows: la 0..lat/2, center offset +H
    for (int la=0;la<=lat/2;++la){float polar=pi*la/lat;
        float sp=std::sin(polar),cp=std::cos(polar);
        for (int lo=0;lo<=lon;++lo){float az=2.0f*pi*lo/lon;
            Vec3 n{sp*std::cos(az),cp,sp*std::sin(az)};
            verts.push_back({{n.x,H+cp,n.z},n,1,1,1,1});}}
    // bottom hemisphere rows: la lat/2..lat, center offset -H (skip equator duplicate)
    for (int la=lat/2+1;la<=lat;++la){float polar=pi*la/lat;
        float sp=std::sin(polar),cp=std::cos(polar);
        for (int lo=0;lo<=lon;++lo){float az=2.0f*pi*lo/lon;
            Vec3 n{sp*std::cos(az),cp,sp*std::sin(az)};
            verts.push_back({{n.x,-H+cp,n.z},n,1,1,1,1});}}
    int topRows=lat/2+1, totRows=lat+1;
    // top hemisphere bands
    for (int la=0;la<topRows-1;++la) for (int lo=0;lo<lon;++lo){
        unsigned a=la*(lon+1)+lo,b=a+lon+1; idx.insert(idx.end(),{a,b,a+1,a+1,b,b+1});}
    // cylinder band (top equator → bottom equator)
    {int la=topRows-1; for (int lo=0;lo<lon;++lo){
        unsigned a=la*(lon+1)+lo, b=(la+1)*(lon+1)+lo;
        idx.insert(idx.end(),{a,b,a+1,a+1,b,b+1});}}
    // bottom hemisphere bands
    for (int la=topRows;la<totRows-1;++la) for (int lo=0;lo<lon;++lo){
        unsigned a=la*(lon+1)+lo,b=a+lon+1; idx.insert(idx.end(),{a,b,a+1,a+1,b,b+1});}
    uploadIndexedMesh(gCapsuleMesh,verts,idx);
}

void createCarMesh() {
    // Clean low-poly car: body slab + cabin with vertical walls.
    // Only axis-aligned normals — safe under non-uniform scale.
    // Glass tint on windows via vertex colour.
    std::vector<Vertex> verts; std::vector<unsigned int> idx;
    auto face=[&](Vec3 n, Vec3 a, Vec3 b, Vec3 c, Vec3 d,
                  float r=1.f,float g=1.f,float bl=1.f){
        unsigned base=(unsigned)verts.size();
        for (const Vec3& p:{a,b,c,d}) verts.push_back({p,n,r,g,bl,1.f});
        idx.insert(idx.end(),{base,base+2,base+1,base+1,base+2,base+3});};

    // ── Body slab ────────────────────────────────────────────────────────────
    const float BY0=-1.f, BY1=0.10f;
    face({ 1,0,0},{ 1,BY0,-1},{ 1,BY0, 1},{ 1,BY1,-1},{ 1,BY1, 1});
    face({-1,0,0},{-1,BY0, 1},{-1,BY0,-1},{-1,BY1, 1},{-1,BY1,-1});
    face({0,-1,0},{-1,BY0,-1},{ 1,BY0,-1},{-1,BY0, 1},{ 1,BY0, 1});
    face({0, 1,0},{-1,BY1, 1},{ 1,BY1, 1},{-1,BY1,-1},{ 1,BY1,-1}); // hood+trunk top
    face({0,0, 1},{ 1,BY0, 1},{-1,BY0, 1},{ 1,BY1, 1},{-1,BY1, 1}); // rear bumper
    face({0,0,-1},{-1,BY0,-1},{ 1,BY0,-1},{-1,BY1,-1},{ 1,BY1,-1}); // front grille

    // ── Cabin ────────────────────────────────────────────────────────────────
    const float CX0=-0.70f, CX1=0.70f;
    const float CY0= 0.10f, CY1=1.00f;
    const float CZ0=-0.62f, CZ1=0.52f;
    const float WY  = CY0+(CY1-CY0)*0.35f; // door-to-window sill height
    const float PW  = 0.12f;               // A/C-pillar width in Z
    const float Gr=0.50f,Gg=0.62f,Gb=0.78f; // glass tint colour

    // Door panels (lower cabin sides, body colour)
    face({ 1,0,0},{CX1,CY0,CZ0},{CX1,CY0,CZ1},{CX1,WY, CZ0},{CX1,WY, CZ1});
    face({-1,0,0},{CX0,CY0,CZ1},{CX0,CY0,CZ0},{CX0,WY, CZ1},{CX0,WY, CZ0});
    // A-pillar (front) and C-pillar (rear) strips above window sill
    face({ 1,0,0},{CX1,WY,CZ0},{CX1,WY,CZ0+PW},{CX1,CY1,CZ0},{CX1,CY1,CZ0+PW});
    face({ 1,0,0},{CX1,WY,CZ1-PW},{CX1,WY,CZ1},{CX1,CY1,CZ1-PW},{CX1,CY1,CZ1});
    face({-1,0,0},{CX0,WY,CZ0+PW},{CX0,WY,CZ0},{CX0,CY1,CZ0+PW},{CX0,CY1,CZ0});
    face({-1,0,0},{CX0,WY,CZ1},{CX0,WY,CZ1-PW},{CX0,CY1,CZ1},{CX0,CY1,CZ1-PW});
    // Side windows (glass tint between pillars)
    face({ 1,0,0},{CX1,WY,CZ0+PW},{CX1,WY,CZ1-PW},{CX1,CY1,CZ0+PW},{CX1,CY1,CZ1-PW},Gr,Gg,Gb);
    face({-1,0,0},{CX0,WY,CZ1-PW},{CX0,WY,CZ0+PW},{CX0,CY1,CZ1-PW},{CX0,CY1,CZ0+PW},Gr,Gg,Gb);
    // Roof
    face({0,1,0},{CX0,CY1,CZ1},{CX1,CY1,CZ1},{CX0,CY1,CZ0},{CX1,CY1,CZ0});
    // Windshield (glass)
    face({0,0,-1},{CX0,CY0,CZ0},{CX1,CY0,CZ0},{CX0,CY1,CZ0},{CX1,CY1,CZ0},Gr,Gg,Gb);
    // Rear window (glass)
    face({0,0, 1},{CX1,CY0,CZ1},{CX0,CY0,CZ1},{CX1,CY1,CZ1},{CX0,CY1,CZ1},Gr,Gg,Gb);

    uploadIndexedMesh(gCarMesh,verts,idx);
}

void updateShadowMesh() {
    constexpr int segments = 64;
    float groundY = groundHeight(ballPosition.x, ballPosition.z);
    float height = std::max(0.0f, ballPosition.y - ballRadius - groundY);
    float radius = ballRadius * (1.8f + height * 0.06f);
    float alpha = 0.22f / (1.0f + height * 0.18f);
    std::vector<Vertex> vertices;
    vertices.push_back({{ballPosition.x, groundY + 0.02f, ballPosition.z}, {}, 0.0f, 0.0f,
                        0.0f, alpha});
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * pi * i / segments;
        float x = ballPosition.x + std::cos(angle) * radius;
        float z = ballPosition.z + std::sin(angle) * radius;
        vertices.push_back({{x, groundHeight(x, z) + 0.02f, z}, {}, 0.0f, 0.0f, 0.0f, 0.0f});
    }
    shadowMesh.mode = GL_TRIANGLE_FAN;
    uploadVertices(shadowMesh, vertices, GL_DYNAMIC_DRAW);
}

void drawMesh(const Mesh& mesh, const Mat4& viewProjection, const Mat4& model,
              bool lit, bool specular = false) {
    Mat4 mvp = multiply(viewProjection, model);
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(modelLocation, 1, GL_FALSE, model.m);
    glUniform1i(litLocation, lit);
    glUniform1i(specularLocation, specular);
    glBindVertexArray(mesh.vao);
    if (mesh.indexed) {
        glDrawElements(mesh.mode, mesh.count, GL_UNSIGNED_INT, nullptr);
    } else {
        glDrawArrays(mesh.mode, 0, mesh.count);
    }
}

// ── Bullet world helpers ──────────────────────────────────────────────────────
float computeMass() {
    float r = ballRadius;
    float v = (objectShape == 0)
        ? (4.0f / 3.0f) * pi * r * r * r
        : 8.0f * r * r * r;
    return kObjectMaterials[objectMaterialIdx].rho * v;
}

#include "sim/physics_math.h"   // computeReducedModulus, computeCriticalFractureVelocity, computeFragmentCount, shapeVolume


struct IdealElasticImpactEstimate {
    float impactEnergyJ = 0.0f;
    float impactSpeedMS = 0.0f;
    float totalCompressionM = 0.0f;
    float maxForceN = 0.0f;
    float maxPressurePa = 0.0f;
    float objectVolumeChangeM3 = 0.0f;
};

// Idealized 1D elastic estimate for flat contact.
// It is exact only inside this simplified model: flat contact area, linear
// elasticity, no plasticity, no heat/sound/fracture, and no wave effects.
IdealElasticImpactEstimate estimateIdealElasticImpact(float dropHeightM, float contactAreaM2) {
    IdealElasticImpactEstimate out{};
    const ObjectMaterial& om = kObjectMaterials[objectMaterialIdx];
    const GroundMaterial& gm = kGroundMaterials[groundMaterialIdx];
    float mass = computeMass();
    float g = kGravities[gravityEnvIdx].g;
    float Eobj = om.E_GPa * 1e9f;
    float Egnd = gm.E_GPa * 1e9f;
    float objectHeight = (objectShape == 0) ? (2.0f * ballRadius) : (2.0f * ballRadius);
    float groundEffectiveHeight = objectHeight; // must be confirmed for a finite target block

    out.impactEnergyJ = mass * g * dropHeightM;
    out.impactSpeedMS = std::sqrt(std::max(0.0f, 2.0f * g * dropHeightM));

    float kObj = Eobj * contactAreaM2 / std::max(objectHeight, 1e-6f);
    float kGnd = Egnd * contactAreaM2 / std::max(groundEffectiveHeight, 1e-6f);
    float kTotal = 1.0f / (1.0f/std::max(kObj,1e-6f) + 1.0f/std::max(kGnd,1e-6f));

    out.totalCompressionM = std::sqrt(2.0f * out.impactEnergyJ / std::max(kTotal, 1e-6f));
    out.maxForceN = kTotal * out.totalCompressionM;
    out.maxPressurePa = out.maxForceN / std::max(contactAreaM2, 1e-6f);

    // Bulk modulus K = E / (3(1 - 2ν)) for isotropic linear elasticity.
    float K = Eobj / std::max(3.0f * (1.0f - 2.0f * om.nu), 1e-6f);
    float volume = (objectShape == 0)
        ? (4.0f / 3.0f) * pi * ballRadius * ballRadius * ballRadius
        : 8.0f * ballRadius * ballRadius * ballRadius;
    out.objectVolumeChangeM3 = -volume * out.maxPressurePa / std::max(K, 1e-6f);
    return out;
}

void destroyBulletWorld() {
    // Remove constraints first (must precede removing rigid bodies)
    for (auto& sc : gSceneConstraints) {
        if (gBtWorld && sc.bt) gBtWorld->removeConstraint(sc.bt);
        delete sc.bt; sc.bt=nullptr;
    }
    gSceneConstraints.clear();

    // Remove scene objects first
    for (auto& obj : gSceneObjects) {
        if (gBtWorld && obj.body) gBtWorld->removeRigidBody(obj.body);
        delete obj.body; delete obj.bms; delete obj.bshape;
        obj.body=nullptr; obj.bms=nullptr; obj.bshape=nullptr;
    }
    gSceneObjects.clear();
    gSelectedObjIdx = -1;

    // Remove and delete fragment bodies first
    for (auto& f : gFragments) {
        if (gBtWorld) gBtWorld->removeRigidBody(f.body);
        delete f.body; delete f.ms; delete f.shape;
    }
    gFragments.clear();

    if (gBtWorld) {
        if (gBtObject) gBtWorld->removeRigidBody(gBtObject);
        if (gBtGround) gBtWorld->removeRigidBody(gBtGround);
    }
    delete gBtObject;   gBtObject   = nullptr;
    delete gBtObjMS;    gBtObjMS    = nullptr;
    delete gBtObjShape; gBtObjShape = nullptr;
    delete gBtGround;   gBtGround   = nullptr;
    delete gBtGndMS;    gBtGndMS    = nullptr;
    delete gBtGndShape; gBtGndShape = nullptr;
    delete gBtMesh;     gBtMesh     = nullptr;
    delete gBtWorld;    gBtWorld    = nullptr;
    delete gBtSolver;   gBtSolver   = nullptr;
    delete gBtBP;       gBtBP       = nullptr;
    delete gBtDispatch; gBtDispatch = nullptr;
    delete gBtConfig;   gBtConfig   = nullptr;
    gObjectFragmented = false;
}

void triggerFracture(Vec3 impactPos, Vec3 impactVel) {
    if (!gBtWorld || gObjectFragmented) return;
    gObjectFragmented = true;
    if (gBtObject) gBtWorld->removeRigidBody(gBtObject);

    // ── Material-science energy budget ────────────────────────────────────────
    const ObjectMaterial& om = kObjectMaterials[objectMaterialIdx];
    const GroundMaterial& gm = kGroundMaterials[groundMaterialIdx];
    float mass     = computeMass();
    float speed    = std::sqrt(dot(impactVel, impactVel));
    float E_impact = 0.5f * mass * speed * speed;

    // Energy bounced back: E_rebound = E_impact × e²_pair
    float e_pair    = om.e_rest * gm.e_rest;
    float E_scatter = E_impact * (1.0f - e_pair * e_pair);

    // ── Grady–Kipp: fragment count ────────────────────────────────────────────
    int nShards = computeFragmentCount(objectMaterialIdx, ballRadius, E_scatter);

    // ── Volume conservation → shard radius ───────────────────────────────────
    float objVol  = (objectShape == 0)
                  ? (4.0f/3.0f * pi * ballRadius*ballRadius*ballRadius)
                  : (8.0f * ballRadius*ballRadius*ballRadius);
    float shardR  = std::cbrt(objVol / nShards / (4.0f/3.0f * pi)) * 1.1f;
    float shardMass = mass / nShards;

    // ── Surface energy consumed by new crack faces (G_c = K_Ic²/E) ───────────
    // Total new surface area ≈ n^(1/3) × original surface
    float K_Ic_Pa  = om.K_Ic  * 1e6f;
    float E1_Pa    = om.E_GPa * 1e9f;
    float G_c      = (K_Ic_Pa * K_Ic_Pa) / std::max(E1_Pa, 1.0f);  // J/m²
    float A_new    = 4.0f * pi * ballRadius * ballRadius * std::cbrt((float)nShards);
    float E_surface = G_c * A_new;
    float E_kinShards = std::max(0.0f, E_scatter - E_surface);
    float vShardMean  = std::sqrt(2.0f * E_kinShards / std::max(mass, 0.001f));

    // ── Scatter direction cone based on ground normal + reflection ────────────
    Vec3 gNorm    = groundNormal(impactPos.x, impactPos.z);
    Vec3 impDir   = normalize(impactVel);
    // Reflect impact direction off ground normal
    Vec3 reflected = normalize(impDir - gNorm * (2.0f * dot(impDir, gNorm)));

    // Cone half-angle: ice=wide (140°), wood=narrower (80°)
    float halfAngle = (objectMaterialIdx == 7) ? (pi * 0.78f) : (pi * 0.44f);

    // Orthonormal basis for cone sampling
    Vec3 bX = (std::fabs(reflected.x) < 0.9f)
              ? normalize(cross(reflected, {1,0,0}))
              : normalize(cross(reflected, {0,1,0}));
    Vec3 bY = cross(reflected, bX);

    // Deterministic, high-quality pseudo-random (Wang hash)
    auto rng = [](int seed) -> float {
        unsigned s = (unsigned)seed * 2654435761u;
        s ^= s >> 16; s *= 0x45d9f3b; s ^= s >> 16;
        return (float)(s & 0xFFFFu) / 65535.0f;
    };

    for (int i = 0; i < nShards; ++i) {
        // ── Direction: uniform sample inside scatter cone ─────────────────────
        float cosMin   = std::cos(halfAngle);
        float cosTheta = cosMin + rng(i*7+1) * (1.0f - cosMin);
        float sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
        float phi      = 2.0f * pi * rng(i*7+2);
        Vec3  dir      = normalize(
            reflected * cosTheta
            + bX * (sinTheta * std::cos(phi))
            + bY * (sinTheta * std::sin(phi)));

        // ── Speed: ±35% around mean, proportional to direction alignment ──────
        float alignment = std::max(0.2f, dot(dir, reflected)); // shards in main dir faster
        float speed_i   = vShardMean * alignment * (0.65f + 0.70f * rng(i*7+3));

        // ── Origin: random point inside original object volume ────────────────
        float ox = (rng(i*7+4) - 0.5f) * ballRadius * 1.0f;
        float oy = rng(i*7+5) * ballRadius * 0.9f;   // bias above impact plane
        float oz = (rng(i*7+6) - 0.5f) * ballRadius * 1.0f;

        // ── Convex hull shard geometry (irregular, not sphere) ────────────────
        btConvexHullShape* hull = new btConvexHullShape();
        float rBase = shardR * (0.72f + 0.40f * rng(i*3));
        for (int p = 0; p < 12; ++p) {
            float ang = p * pi * 0.7391f + i * 0.4f;
            float el  = p * pi * 0.3927f;
            float rr  = rBase * (0.50f + 0.50f * rng(p*7 + i));
            float ry  = rBase * (0.55f + 0.45f * rng(p*13 + i + 1));
            hull->addPoint(btVector3(rr*std::cos(ang)*std::cos(el),
                                     ry*std::sin(el),
                                     rr*std::sin(ang)*std::cos(el)));
        }
        hull->recalcLocalAabb();

        btVector3 inertia(0,0,0);
        hull->calculateLocalInertia(shardMass, inertia);

        btTransform t; t.setIdentity();
        t.setOrigin(btVector3(impactPos.x + ox,
                              std::max(impactPos.y, groundHeight(impactPos.x, impactPos.z))
                              + oy + shardR,
                              impactPos.z + oz));

        btDefaultMotionState* ms = new btDefaultMotionState(t);
        btRigidBody::btRigidBodyConstructionInfo ci(shardMass, ms, hull, inertia);
        ci.m_friction       = om.mu_fric;
        ci.m_restitution    = om.e_rest * gm.e_rest * 0.6f; // shards absorb more
        ci.m_linearDamping  = 0.0f;
        ci.m_angularDamping = 0.0f;
        btRigidBody* body = new btRigidBody(ci);
        body->setCcdMotionThreshold(0.01f * shardR);
        body->setCcdSweptSphereRadius(0.12f * shardR);

        body->setLinearVelocity(btVector3(dir.x*speed_i, dir.y*speed_i, dir.z*speed_i));
        // Angular spin: proportional to tangential speed (tumbles faster when spinning sideways)
        float spin = speed_i * 3.0f / std::max(shardR, 0.01f);
        body->setAngularVelocity(btVector3(
            (rng(i*7+4) - 0.5f) * spin,
            (rng(i*7+5) - 0.5f) * spin,
            (rng(i*7+6) - 0.5f) * spin));
        body->setActivationState(DISABLE_DEACTIVATION);
        gBtWorld->addRigidBody(body);

        float sx = 0.6f + 0.7f * rng(i*17);
        float sy = 0.5f + 0.8f * rng(i*19+1);
        float sz = 0.6f + 0.7f * rng(i*23+2);
        gFragments.push_back({body, hull, ms, sx, sy, sz});
    }
}

// Build the correct Bullet collision shape for a scene object, properly encoding
// non-uniform scale into shape geometry rather than relying on setLocalScaling.
//
// Reason: btSphereShape is a "margin-only" shape — setLocalScaling only scales
// the X axis (getRadius = r*sx), ignoring sy and sz entirely. Non-uniform sphere
// scales must be encoded as a convex hull ellipsoid instead.
// btBoxShape and btCylinderShape accept half-extents directly, so no scaling needed.
static btCollisionShape* makeBtShape(int shapeType, float r, float sx, float sy, float sz) {
    bool uniform = (sx == sy && sy == sz);
    switch (shapeType) {
        case 0: // Sphere
            if (uniform) return new btSphereShape(r * sx);
            {   // Non-uniform: sample ellipsoid into a convex hull
                auto* hull = new btConvexHullShape();
                float ax=r*sx, ay=r*sy, az=r*sz;
                const int latS=8, lonS=12;
                for (int la=0; la<=latS; ++la) {
                    float polar=pi*la/latS, sp=std::sin(polar), cp=std::cos(polar);
                    for (int lo=0; lo<=lonS; ++lo) {
                        float azm=2.f*pi*lo/lonS;
                        hull->addPoint(btVector3(ax*sp*std::cos(azm), ay*cp, az*sp*std::sin(azm)));
                    }
                }
                hull->recalcLocalAabb();
                return hull;
            }
        case 1: { // Box — half-extents encode scale directly
            auto* s = new btBoxShape(btVector3(r*sx, r*sy, r*sz));
            s->setMargin(0.005f); // small margin so visual mesh sits flush on ground
            return s; }
        case 2: // Cylinder
            return new btCylinderShape(btVector3(r*sx, r*sy, r*sz));
        case 3: // Cone — base radius = XZ avg, height = Y
            return new btConeShape(r * (sx+sz)*0.5f, 2.0f*r*sy);
        case 4: // Capsule — radius = XZ avg, cylindrical section height = r*sy
            return new btCapsuleShape(r * (sx+sz)*0.5f, r*sy);
        case 5: { // Car — box shape with given half-extents
            auto* s = new btBoxShape(btVector3(r*sx, r*sy, r*sz));
            s->setMargin(0.005f);
            return s; }
        default:
            return new btSphereShape(r);
    }
}

// spawnPos.y == -1e30f is a sentinel meaning "auto-height above ground"
static constexpr float kAutoSpawnY = -1e30f;

void addSceneObject(int shapeType, int matIdx, float r,
                    float sx=1.f, float sy=1.f, float sz=1.f,
                    Vec3 spawnPos={0, kAutoSpawnY, 0},
                    Quat spawnOrient={1,0,0,0}) {
    if (!gBtWorld) return;
    SceneObject obj;
    obj.shapeType = shapeType; obj.matIdx = matIdx; obj.r = r;
    obj.sx=sx; obj.sy=sy; obj.sz=sz;
    if (spawnPos.y <= kAutoSpawnY + 1.0f) {
        float halfH = r * sy;
        // Spread each new object in a golden-angle sunflower spiral so no two
        // objects share the same XZ position.  Overlapping spawns lock Bullet's
        // contact solver: forces cancel and objects never fall.
        int   n     = (int)gSceneObjects.size();
        float angle = (float)n * 2.399963f;           // golden angle ~137.5°
        float dist  = (n == 0) ? 0.0f
                               : (1.5f + (float)n * 0.8f * std::max(r, 0.3f));
        dist = std::min(dist, 30.0f);                 // keep objects in view
        float cx = std::cos(angle) * dist;
        float cz = std::sin(angle) * dist;
        obj.pos    = {cx, groundHeight(cx, cz) + halfH + 2.0f, cz};
        obj.orient = {1,0,0,0};
    } else {
        obj.pos    = spawnPos;
        obj.orient = spawnOrient;
    }
    obj.euler = {0,0,0};
    std::snprintf(obj.label, sizeof(obj.label), "%s #%d",
                  kSceneShapeNames[shapeType], (int)gSceneObjects.size()+1);

    btCollisionShape* cs = makeBtShape(shapeType, r, sx, sy, sz);

    const ObjectMaterial& om = kObjectMaterials[matIdx];
    float mass = om.rho * shapeVolume(shapeType, r, sx, sy, sz);
    btVector3 inertia(0,0,0); cs->calculateLocalInertia(mass, inertia);
    btTransform t; t.setIdentity();
    t.setOrigin(btVector3(obj.pos.x, obj.pos.y, obj.pos.z));
    btDefaultMotionState* ms = new btDefaultMotionState(t);
    btRigidBody::btRigidBodyConstructionInfo ci(mass, ms, cs, inertia);
    ci.m_friction    = om.mu_fric;
    ci.m_restitution = om.e_rest;
    btRigidBody* body = new btRigidBody(ci);
    body->setRollingFriction(om.mu_roll);
    body->setSpinningFriction(om.mu_roll * 0.5f);
    body->setActivationState(DISABLE_DEACTIVATION);
    // CCD: threshold << smallest dimension so tunnelling is caught at any impact speed
    float minDim = r * std::min({sx, sy, sz});
    body->setCcdMotionThreshold(0.01f * minDim);
    body->setCcdSweptSphereRadius(0.15f * minDim);
    gBtWorld->addRigidBody(body);
    obj.body = body; obj.bshape = cs; obj.bms = ms;
    gSceneObjects.push_back(obj);
    gSelectedObjIdx = (int)gSceneObjects.size() - 1;
}

void removeSceneObject(int idx) {
    if (idx < 0 || idx >= (int)gSceneObjects.size()) return;
    // Remove constraints referencing this object first
    for (int ci = (int)gSceneConstraints.size()-1; ci >= 0; --ci) {
        auto& sc = gSceneConstraints[ci];
        if (sc.objA == idx || sc.objB == idx) {
            if (gBtWorld && sc.bt) { gBtWorld->removeConstraint(sc.bt); delete sc.bt; sc.bt=nullptr; }
            gSceneConstraints.erase(gSceneConstraints.begin()+ci);
        } else {
            if (sc.objA > idx) --sc.objA;
            if (sc.objB > idx) --sc.objB;
        }
    }
    auto& obj = gSceneObjects[idx];
    if (gBtWorld && obj.body) gBtWorld->removeRigidBody(obj.body);
    delete obj.body; delete obj.bms; delete obj.bshape;
    gSceneObjects.erase(gSceneObjects.begin() + idx);
    gSelectedObjIdx = std::min(gSelectedObjIdx, (int)gSceneObjects.size() - 1);
    if (gWsConnA == idx) gWsConnA=-1; else if (gWsConnA > idx) --gWsConnA;
    if (gWsConnB == idx) gWsConnB=-1; else if (gWsConnB > idx) --gWsConnB;
}

void addConstraint(SceneConstraint sc) {
    if (!gBtWorld) return;
    bool vA = sc.objA>=0 && sc.objA<(int)gSceneObjects.size() && gSceneObjects[sc.objA].body;
    bool vB = sc.objB>=0 && sc.objB<(int)gSceneObjects.size() && gSceneObjects[sc.objB].body;
    if (!vA) return;
    btRigidBody* ba = gSceneObjects[sc.objA].body;
    btRigidBody* bb = vB ? gSceneObjects[sc.objB].body : nullptr;
    btVector3 pA(sc.pivotA.x,sc.pivotA.y,sc.pivotA.z);
    btVector3 pB(sc.pivotB.x,sc.pivotB.y,sc.pivotB.z);
    btTransform tA,tB; tA.setIdentity(); tB.setIdentity();
    tA.setOrigin(pA); tB.setOrigin(pB);

    btTypedConstraint* c = nullptr;
    switch (sc.typeIdx) {
        case 0: // Point-to-Point
            if (bb) c = new btPoint2PointConstraint(*ba,*bb, pA,pB);
            else    c = new btPoint2PointConstraint(*ba, pA);
            break;
        case 1: { // Hinge
            // Convert world-space hinge axis to each body's local space
            float nx=sc.hingeAxis.x, ny=sc.hingeAxis.y, nz=sc.hingeAxis.z;
            float len=std::sqrt(nx*nx+ny*ny+nz*nz);
            if (len < 1e-6f) { nx=0; ny=1; nz=0; } else { nx/=len; ny/=len; nz/=len; }
            btVector3 worldAxis(nx, ny, nz);
            btVector3 localAxisA = ba->getCenterOfMassTransform().getBasis().transpose() * worldAxis;
            // Build a frame whose Z-axis is the hinge axis
            auto makeHingeFrame = [](btVector3 origin, btVector3 axisZ) {
                axisZ.normalize();
                btVector3 axisX = (std::fabs(axisZ.x()) > 0.9f)
                    ? btVector3(0,1,0).cross(axisZ) : btVector3(1,0,0).cross(axisZ);
                axisX.normalize();
                btVector3 axisY = axisZ.cross(axisX);
                btTransform f; f.setIdentity(); f.setOrigin(origin);
                f.getBasis().setValue(axisX.x(),axisY.x(),axisZ.x(),
                                     axisX.y(),axisY.y(),axisZ.y(),
                                     axisX.z(),axisY.z(),axisZ.z());
                return f;
            };
            btTransform fA = makeHingeFrame(pA, localAxisA);
            if (bb) {
                btVector3 localAxisB = bb->getCenterOfMassTransform().getBasis().transpose() * worldAxis;
                btTransform fB = makeHingeFrame(pB, localAxisB);
                c = new btHingeConstraint(*ba, *bb, fA, fB);
            } else {
                c = new btHingeConstraint(*ba, fA);
            }
            auto* h = static_cast<btHingeConstraint*>(c);
            h->setLimit(sc.limitLow*pi/180.f, sc.limitHigh*pi/180.f);
            break;
        }
        case 2: // Spring
            if (bb) {
                auto* s = new btGeneric6DofSpring2Constraint(*ba,*bb,tA,tB);
                for (int i=0;i<6;i++){ s->enableSpring(i,true); s->setStiffness(i,sc.springK); s->setDamping(i,sc.springD); }
                c = s;
            }
            break;
        case 3: // Fixed
            if (bb) c = new btFixedConstraint(*ba,*bb,tA,tB);
            break;
    }
    if (c) { gBtWorld->addConstraint(c,true); sc.bt=c; }
    if (!sc.label[0])
        std::snprintf(sc.label,sizeof(sc.label),"%s A%d-B%d",
                      kConstraintNames[sc.typeIdx], sc.objA, sc.objB);
    gSceneConstraints.push_back(sc);
}

void removeConstraint(int idx) {
    if (idx<0||idx>=(int)gSceneConstraints.size()) return;
    auto& sc=gSceneConstraints[idx];
    if (gBtWorld&&sc.bt){gBtWorld->removeConstraint(sc.bt);delete sc.bt;sc.bt=nullptr;}
    gSceneConstraints.erase(gSceneConstraints.begin()+idx);
}

void syncSceneObjectFromBullet(SceneObject& obj) {
    if (!obj.body) return;
    btTransform t; obj.body->getMotionState()->getWorldTransform(t);
    const btVector3& o = t.getOrigin();
    obj.pos = {o.x(), o.y(), o.z()};
    btQuaternion q = t.getRotation();
    obj.orient = {q.w(), q.x(), q.y(), q.z()};
    btVector3 lv = obj.body->getLinearVelocity();
    obj.vel = {lv.x(), lv.y(), lv.z()};
}

void pushSceneObjectToBullet(SceneObject& obj) {
    if (!obj.body) return;
    btTransform t; t.setIdentity();
    t.setOrigin(btVector3(obj.pos.x, obj.pos.y, obj.pos.z));
    t.setRotation(btQuaternion(obj.orient.x, obj.orient.y, obj.orient.z, obj.orient.w));
    obj.body->setWorldTransform(t);
    obj.body->getMotionState()->setWorldTransform(t);
    obj.body->setLinearVelocity(btVector3(0,0,0));
    obj.body->setAngularVelocity(btVector3(0,0,0));
    obj.body->clearForces(); obj.body->activate();
}

void rebuildSceneObjectShape(int idx) {
    if (idx < 0 || idx >= (int)gSceneObjects.size() || !gBtWorld) return;
    SceneObject& obj = gSceneObjects[idx];
    // Grab current world transform before destroying
    btTransform cur; obj.body->getMotionState()->getWorldTransform(cur);
    if (gBtWorld && obj.body) gBtWorld->removeRigidBody(obj.body);
    delete obj.body; delete obj.bms; delete obj.bshape;
    obj.body=nullptr; obj.bms=nullptr; obj.bshape=nullptr;

    float r = obj.r;
    btCollisionShape* cs = makeBtShape(obj.shapeType, r, obj.sx, obj.sy, obj.sz);
    const ObjectMaterial& om = kObjectMaterials[obj.matIdx];
    float mass = om.rho * shapeVolume(obj.shapeType, r, obj.sx, obj.sy, obj.sz);
    btVector3 inertia(0,0,0); cs->calculateLocalInertia(mass, inertia);
    btDefaultMotionState* ms = new btDefaultMotionState(cur);
    btRigidBody::btRigidBodyConstructionInfo ci(mass, ms, cs, inertia);
    ci.m_friction = om.mu_fric; ci.m_restitution = om.e_rest;
    btRigidBody* body = new btRigidBody(ci);
    body->setRollingFriction(om.mu_roll);
    body->setSpinningFriction(om.mu_roll * 0.5f);
    body->setActivationState(DISABLE_DEACTIVATION);
    float minD = r * std::min({obj.sx, obj.sy, obj.sz});
    body->setCcdMotionThreshold(0.01f*minD); body->setCcdSweptSphereRadius(0.15f*minD);
    gBtWorld->addRigidBody(body);
    obj.body=body; obj.bshape=cs; obj.bms=ms;
}

struct SceneObjectSnap {
    int shapeType, matIdx;
    float r, sx, sy, sz;
    Vec3 pos; Quat orient;
    char label[32];
};

void buildBulletWorld() {
    gTerrainCX = 0.0f;
    gTerrainCZ = 0.0f;

    // Snapshot scene objects and constraints so they survive the world rebuild
    std::vector<SceneObjectSnap> savedObjs;
    for (auto& obj : gSceneObjects) {
        syncSceneObjectFromBullet(obj);
        SceneObjectSnap snap{};
        snap.shapeType=obj.shapeType; snap.matIdx=obj.matIdx;
        snap.r=obj.r; snap.sx=obj.sx; snap.sy=obj.sy; snap.sz=obj.sz;
        snap.pos=obj.pos; snap.orient=obj.orient;
        std::memcpy(snap.label, obj.label, 32);
        savedObjs.push_back(snap);
    }
    std::vector<SceneConstraint> savedCons;
    for (auto& sc : gSceneConstraints) {
        SceneConstraint copy = sc;
        copy.bt = nullptr;
        savedCons.push_back(copy);
    }
    int prevSel = gSelectedObjIdx;

    destroyBulletWorld();

    gBtConfig   = new btDefaultCollisionConfiguration();
    gBtDispatch = new btCollisionDispatcher(gBtConfig);
    gBtBP       = new btDbvtBroadphase();
    gBtSolver   = new btSequentialImpulseConstraintSolver();
    gBtWorld    = new btDiscreteDynamicsWorld(gBtDispatch, gBtBP, gBtSolver, gBtConfig);
    gBtWorld->setGravity(btVector3(0, -kGravities[gravityEnvIdx].g, 0));
    // More iterations → heavier/larger objects can't compress the contact and sink.
    // Default 10 is too few; heavy cars visibly settle into the ground without this.
    gBtWorld->getSolverInfo().m_numIterations = 50;
    gBtWorld->getSolverInfo().m_erp           = 0.8f;  // push out 80% of penetration per step

    // ── Static terrain ────────────────────────────────────────────────────────
    // Flat ground uses an exact infinite plane. Bumpy terrain uses a high-res
    // mesh (80-unit half-extent, ~1.67 m/cell) centered on the current terrain
    // center (gTerrainCX/CZ), which follows the ball for truly unlimited physics.
    const int   steps  = 96;
    const float extent = 80.0f;
    if (groundType == 0) {
        gBtGndShape = new btStaticPlaneShape(btVector3(0, 1, 0), 0);
    } else {
        float cx = gTerrainCX, cz = gTerrainCZ;
        gBtMesh = new btTriangleMesh();
        for (int zi = 0; zi < steps; ++zi) {
            for (int xi = 0; xi < steps; ++xi) {
                float x0 = cx + (-extent + 2*extent*xi      / steps);
                float z0 = cz + (-extent + 2*extent*zi      / steps);
                float x1 = cx + (-extent + 2*extent*(xi+1)  / steps);
                float z1 = cz + (-extent + 2*extent*(zi+1)  / steps);
                btVector3 a(x0, groundHeight(x0,z0), z0);
                btVector3 b(x1, groundHeight(x1,z0), z0);
                btVector3 c(x0, groundHeight(x0,z1), z1);
                btVector3 d(x1, groundHeight(x1,z1), z1);
                gBtMesh->addTriangle(a, c, b);
                gBtMesh->addTriangle(b, c, d);
            }
        }
        gBtGndShape = new btBvhTriangleMeshShape(gBtMesh, true);
    }
    btTransform groundT; groundT.setIdentity();
    gBtGndMS = new btDefaultMotionState(groundT);
    const GroundMaterial& gm = kGroundMaterials[groundMaterialIdx];
    btRigidBody::btRigidBodyConstructionInfo gCI(0, gBtGndMS, gBtGndShape);
    gCI.m_friction    = gm.mu_fric;
    gCI.m_restitution = gm.e_rest;
    gBtGround = new btRigidBody(gCI);
    gBtGround->setRollingFriction(gm.mu_roll);
    gBtWorld->addRigidBody(gBtGround);

    // ── Restore scene objects and constraints ─────────────────────────────────
    for (auto& snap : savedObjs) {
        addSceneObject(snap.shapeType, snap.matIdx, snap.r,
                       snap.sx, snap.sy, snap.sz, snap.pos, snap.orient);
        if (!gSceneObjects.empty())
            std::memcpy(gSceneObjects.back().label, snap.label, 32);
    }
    for (auto& sc : savedCons)
        addConstraint(sc);
    gSelectedObjIdx = std::min(prevSel, (int)gSceneObjects.size() - 1);
}

// ── Rebuild terrain only (after crater deformation) ──────────────────────────
void rebuildBulletTerrain() {
    if (!gBtWorld) return;
    gBtWorld->removeRigidBody(gBtGround);
    delete gBtGround;   gBtGround   = nullptr;
    delete gBtGndMS;    gBtGndMS    = nullptr;
    delete gBtGndShape; gBtGndShape = nullptr;
    delete gBtMesh;     gBtMesh     = nullptr;

    const int   steps  = 96;
    const float extent = 80.0f;
    float cx = gTerrainCX, cz = gTerrainCZ;
    gBtMesh = new btTriangleMesh();
    for (int zi = 0; zi < steps; ++zi) {
        for (int xi = 0; xi < steps; ++xi) {
            float x0 = cx + (-extent + 2*extent*xi     /steps);
            float z0 = cz + (-extent + 2*extent*zi     /steps);
            float x1 = cx + (-extent + 2*extent*(xi+1) /steps);
            float z1 = cz + (-extent + 2*extent*(zi+1) /steps);
            btVector3 a(x0,groundHeight(x0,z0),z0), b(x1,groundHeight(x1,z0),z0);
            btVector3 c(x0,groundHeight(x0,z1),z1), d(x1,groundHeight(x1,z1),z1);
            gBtMesh->addTriangle(a,c,b); gBtMesh->addTriangle(b,c,d);
        }
    }
    gBtGndShape = new btBvhTriangleMeshShape(gBtMesh, true);
    btTransform gt; gt.setIdentity();
    gBtGndMS = new btDefaultMotionState(gt);
    const GroundMaterial& gm = kGroundMaterials[groundMaterialIdx];
    btRigidBody::btRigidBodyConstructionInfo gCI(0, gBtGndMS, gBtGndShape);
    gCI.m_friction    = gm.mu_fric;
    gCI.m_restitution = gm.e_rest;
    gBtGround = new btRigidBody(gCI);
    gBtGround->setRollingFriction(gm.mu_roll);
    gBtWorld->addRigidBody(gBtGround);
    createTerrainMesh(); createGridMesh();  // sync visual mesh
}

// ── Aerodynamic drag + Magnus effect + buoyancy ───────────────────────────────
// Drag coefficient Cd(Re) curve (sphere):
//   Re < 1            → Stokes:  Cd = 24/Re
//   1 ≤ Re < 1000     → Schiller–Naumann: Cd ≈ 24/Re·(1+0.15 Re^0.687)
//   1000 ≤ Re < 2×10⁵ → Newton plateau: Cd = 0.44  (sphere) / 1.05 (box)
//   Re ≥ 2×10⁵        → Supercritical (drag crisis): Cd ≈ 0.10  (sphere)
// For a box Cd is ~2× sphere, no drag crisis.
void applyAerodynamicForces() {
    if (!gBtObject || gObjectFragmented) return;
    const AtmosphereType& atm = kAtmospheres[atmosphereIdx];
    gAeroDragN = gAeroMagnusN = 0.0f;
    if (atm.rho < 1e-8f) return;

    btVector3 vel    = gBtObject->getLinearVelocity();
    btVector3 angVel = gBtObject->getAngularVelocity();
    float v2 = vel.length2();
    if (v2 < 1e-6f) return;
    float v  = std::sqrt(v2);

    float r    = ballRadius;
    float D    = 2.0f * r;                    // characteristic diameter
    float area = (objectShape == 0) ? pi*r*r : 4.0f*r*r;

    // Reynolds number — fundamental dimensionless flow parameter
    float Re = (atm.mu > 1e-12f) ? (atm.rho * v * D / atm.mu) : 1e6f;

    float Cd;
    if (objectShape == 1) {
        // Box: no drag crisis, Cd roughly constant at high Re
        Cd = (Re < 1.0f)    ? 24.0f / std::max(Re, 1e-6f)
           : (Re < 1000.0f) ? (24.0f/Re) * (1.0f + 0.15f*std::pow(Re, 0.687f))
                             : 1.05f;
    } else {
        // Sphere: Cd(Re) with drag crisis at Re≈2×10⁵
        if      (Re < 1.0f)    Cd = 24.0f / std::max(Re, 1e-6f);
        else if (Re < 1000.0f) Cd = (24.0f/Re) * (1.0f + 0.15f*std::pow(Re, 0.687f));
        else if (Re < 2.0e5f)  Cd = 0.44f;
        else                   Cd = 0.10f;  // supercritical — sudden drop (Eiffel effect)
    }

    // F_drag = −½ρ·Cd·A·v²·v̂
    float dragMag = 0.5f * atm.rho * Cd * area * v2;
    gBtObject->applyForce(-(vel / v) * dragMag, btVector3(0,0,0));
    gAeroDragN = dragMag;

    // F_Magnus = ρ·V·(ω × v)  — spinning object deflects in fluid (gyroscopic)
    float volume = (objectShape == 0) ? (4.0f/3.0f)*pi*r*r*r : 8.0f*r*r*r;
    btVector3 magnus = angVel.cross(vel) * (atm.rho * volume * 0.5f);
    gBtObject->applyForce(magnus, btVector3(0,0,0));
    gAeroMagnusN = magnus.length();

    // Buoyancy: F_b = ρ_fluid·V·g (upward) — significant in water / Venus
    float g = kGravities[gravityEnvIdx].g;
    gBtObject->applyForce(btVector3(0, atm.rho * volume * g, 0), btVector3(0,0,0));
}

// ── Contact manifold inspection ───────────────────────────────────────────────
// Called AFTER stepSimulation so impulse values are populated.
// Handles: fracture trigger, terrain deformation, impact flash.
// Fracture check uses v_c from Hertz contact theory + Griffith criterion.
// Terrain deformation uses Terzaghi bearing capacity, not impulse heuristic.
void checkContactManifolds() {
    if (!gBtWorld || !gBtObject || gObjectFragmented) return;
    float mass = computeMass();
    float v_c  = computeCriticalFractureVelocity(
                     objectMaterialIdx, groundMaterialIdx, ballRadius);

    int nm = gBtWorld->getDispatcher()->getNumManifolds();
    for (int i = 0; i < nm; ++i) {
        btPersistentManifold* mf = gBtWorld->getDispatcher()->getManifoldByIndexInternal(i);
        bool hasObj = (mf->getBody0() == gBtObject || mf->getBody1() == gBtObject);
        if (!hasObj) continue;

        for (int j = 0; j < mf->getNumContacts(); ++j) {
            const btManifoldPoint& pt = mf->getContactPoint(j);
            float imp = pt.m_appliedImpulse;
            if (imp < 0.05f) continue;

            btVector3 cp = (mf->getBody0() == gBtObject)
                           ? pt.getPositionWorldOnA()
                           : pt.getPositionWorldOnB();
            gFlashPos     = {cp.x(), cp.y(), cp.z()};
            gFlashTimer   = 0.22f;
            gFlashImpulse = imp;

            // Normal impact speed should come from the pre-contact velocity,
            // not from Bullet impulse. J/m is a velocity change caused by the
            // solver, and depends on timestep/restitution/contact constraints.
            btVector3 nb = pt.m_normalWorldOnB;
            Vec3 normal{nb.x(), nb.y(), nb.z()};
            float v_n = std::fabs(dot(gPrevVelocity, normal));

            // ── Hertz + Griffith-inspired fracture criterion ──────────────────
            if (v_c > 0.0f && v_n >= v_c) {
                gLastImpactVel = v_n;
                triggerFracture(gFlashPos, gPrevVelocity);
                return;
            }

            // ── Terzaghi bearing capacity (soil / sand deformation) ───────────
            // Contact area a ≈ Hertz radius: a = (3FR/4E*)^(1/3) ≈ 0.3·R
            bool softGround = (groundMaterialIdx == 2 || groundMaterialIdx == 5);
            if (softGround) {
                float R     = ballRadius;
                // Do not infer force/pressure from impulse unless contact duration
                // has been confirmed or calibrated. F_avg = J / Δt.
                if (gContactDurationEstimate <= 0.0f) continue;
                float contactA = pi * 0.09f * R * R; // model assumption: effective contact patch ≈ 0.3R radius
                float forceAvg = imp / gContactDurationEstimate;
                float press    = forceAvg / (contactA + 1e-8f);
                // Terzaghi ultimate bearing capacity: q_ult = c·Nc + γ·D·Nq.
                // Approximate shallow-footing values (embedment D ~ ball radius):
                //   Sand (loose dry, φ≈30°, c=0): Nq≈18, γ≈16 kN/m³, D≈r → ~75 kPa
                //   Soil (medium dry, c≈20 kPa, φ≈25°): Nc≈20 → c·Nc + γDNq ≈ 150 kPa
                // These are single-point estimates; a rigorous model needs measured φ,c,γ,D.
                float q_ult_Pa = (groundMaterialIdx == 5) ? 75000.f : 150000.f;  // sand : soil
                if (press > q_ult_Pa) {
                    float overpressure = press / q_ult_Pa - 1.0f;
                    float scale   = std::min(1.0f, overpressure * 0.2f);
                    gCraters.push_back({cp.x(), cp.z(),
                        R * (0.18f + 0.40f * scale),   // depth
                        R * (1.2f  + 1.0f  * scale)});  // radius
                    gTerrainDirty = true;
                }
            }
        }
    }
}

void resetBall() {
    // Clear craters and reset terrain to world origin for a clean drop
    gCraters.clear();
    gTerrainDirty = false;
    gTerrainCX = 0.0f;
    gTerrainCZ = 0.0f;
    createTerrainMesh(); createGridMesh();

    gTrail.clear();
    gFlashTimer   = 0.0f;
    gFlashImpulse = 0.0f;
    gAeroDragN    = 0.0f;
    gAeroMagnusN  = 0.0f;
    gSimTime      = 0.0f;
    gTimeSeries.clear();
    gImpactSnap   = ImpactSnapshot{};

    float spawnY  = groundHeight(0,0) + ballDropHeight + ballRadius;
    ballPosition      = {0.0f, spawnY, 0.0f};
    ballVelocity      = {0.0f, 0.0f, 0.0f};
    objectOrientation = {1.0f, 0.0f, 0.0f, 0.0f};
    angularVelocity   = {0.0f, 0.0f, 0.0f};
    gLastImpactVel    = 0.0f;
    gPrevVelocity     = {0.0f, 0.0f, 0.0f};

    for (auto& f : gFragments) {
        if (gBtWorld) gBtWorld->removeRigidBody(f.body);
        delete f.body; delete f.ms; delete f.shape;
    }
    gFragments.clear();
    gObjectFragmented = false;

    if (gBtObject) {
        if (gBtWorld->getCollisionObjectArray().findLinearSearch(gBtObject)
                == gBtWorld->getNumCollisionObjects())
            gBtWorld->addRigidBody(gBtObject);
        btTransform t; t.setIdentity();
        t.setOrigin(btVector3(0.0f, spawnY, 0.0f));
        gBtObject->setWorldTransform(t);
        gBtObject->getMotionState()->setWorldTransform(t);
        gBtObject->setLinearVelocity(btVector3(0,0,0));
        gBtObject->setAngularVelocity(btVector3(0,0,0));
        gBtObject->clearForces();
        gBtObject->activate();
    }
    if (gBtWorld && groundType != 0) rebuildBulletTerrain();
}

// ── Sphere physics (point-contact, existing model) ────────────────────────────
void updatePhysicsSphere(float dt) {
    float remaining = std::min(dt, 0.05f);
    while (remaining > 0.0f) {
        float step   = std::min(remaining, 0.008f);
        remaining   -= step;
        ballVelocity.y += gravity * step;
        ballPosition    = ballPosition + ballVelocity * step;
        float groundY   = groundHeight(ballPosition.x, ballPosition.z) + ballRadius;
        if (ballPosition.y <= groundY) {
            ballPosition.y = groundY;
            Vec3  n  = groundNormal(ballPosition.x, ballPosition.z);
            float ns = dot(ballVelocity, n);
            if (ns < 0.0f)
                ballVelocity = ballVelocity - n * ((1.0f + bounceFactor) * ns);
            Vec3  tang = ballVelocity - n * dot(ballVelocity, n);
            float ts   = std::sqrt(dot(tang, tang));
            if (ts > 0.001f) {
                float imp = std::min(friction * 9.8f * step, ts);
                ballVelocity = ballVelocity - tang * (imp / ts);
            }
        }
    }
}

// ── Box rigid-body physics (corner contacts + angular impulse) ────────────────
void updatePhysicsBox(float dt) {
    constexpr float mass  = 1.0f;
    // Moment of inertia for a solid cube of side 2r: I = (2/3)*m*r^2
    const float I_inv = 1.5f / (mass * ballRadius * ballRadius);

    float remaining = std::min(dt, 0.05f);
    while (remaining > 0.0f) {
        float step   = std::min(remaining, 0.008f);
        remaining   -= step;

        // Integrate linear
        ballVelocity.y += gravity * step;
        ballPosition    = ballPosition + ballVelocity * step;

        // Integrate angular into orientation  (dq = 0.5 * q * ω·dt)
        Quat dq = qMul(objectOrientation,
                       {0.0f,
                        angularVelocity.x * step * 0.5f,
                        angularVelocity.y * step * 0.5f,
                        angularVelocity.z * step * 0.5f});
        objectOrientation = qNorm({objectOrientation.w + dq.w,
                                   objectOrientation.x + dq.x,
                                   objectOrientation.y + dq.y,
                                   objectOrientation.z + dq.z});

        // Test all 8 corners against terrain
        const float r = ballRadius;
        const Vec3 localCorners[8] = {
            {-r,-r,-r},{r,-r,-r},{-r,r,-r},{r,r,-r},
            {-r,-r, r},{r,-r, r},{-r,r, r},{r,r, r}
        };

        for (const Vec3& lc : localCorners) {
            Vec3  rc        = qRot(objectOrientation, lc);   // corner relative to CoM
            Vec3  wc        = ballPosition + rc;             // world position of corner
            float groundY   = groundHeight(wc.x, wc.z);
            float pen       = groundY - wc.y;
            if (pen <= 0.0f) continue;

            // Push CoM up to resolve penetration
            ballPosition.y += pen;
            wc.y            = groundY;

            Vec3  n        = groundNormal(wc.x, wc.z);
            Vec3  vContact = ballVelocity + cross(angularVelocity, rc);
            float vn       = dot(vContact, n);
            if (vn >= 0.0f) continue;   // already separating

            // Normal impulse:  J = -(1+e)*vn / (1/m + n·(I⁻¹(r×n))×r)
            float rcn_sq = dot(rc, rc) - dot(rc, n) * dot(rc, n);
            float denom  = 1.0f / mass + I_inv * rcn_sq;
            float Jn     = -(1.0f + bounceFactor) * vn / denom;

            ballVelocity    = ballVelocity + n * (Jn / mass);
            angularVelocity = angularVelocity + cross(rc, n * Jn) * I_inv;

            // Friction impulse (tangential)
            Vec3  vt    = vContact - n * vn;
            float vtLen = std::sqrt(dot(vt, vt));
            if (vtLen > 0.001f) {
                Vec3  tDir   = vt * (1.0f / vtLen);
                Vec3  rct    = cross(rc, tDir);
                float denomT = 1.0f / mass + I_inv * dot(rct, rct);
                float Jt     = std::min(friction * std::fabs(Jn),
                                        vtLen / denomT);
                ballVelocity    = ballVelocity - tDir * (Jt / mass);
                angularVelocity = angularVelocity - cross(rc, tDir * Jt) * I_inv;
            }
        }

        // Light angular damping (air drag)
        angularVelocity = angularVelocity * (1.0f - 0.02f * step);
    }
}

void updatePhysics(float dt) {
    if (gBtWorld) {
        applyAerodynamicForces();

        for (auto& obj : gSceneObjects) {
            if (obj.forceOn && obj.forceMag > 1e-4f && obj.body) {
                Vec3 fd = normalize(obj.forceDir);
                obj.body->applyForce(
                    btVector3(fd.x*obj.forceMag, fd.y*obj.forceMag, fd.z*obj.forceMag),
                    btVector3(0,0,0));
            }
        }

        float safeDt = std::min(dt, 0.05f);
        gBtWorld->stepSimulation(safeDt, 12, 1.0f / 120.0f);

        if (!gObjectFragmented && gBtObject) {
            btTransform t;
            gBtObject->getMotionState()->getWorldTransform(t);
            const btVector3& o = t.getOrigin();
            ballPosition = {o.x(), o.y(), o.z()};
            btQuaternion q = t.getRotation();
            objectOrientation = {q.w(), q.x(), q.y(), q.z()};
            btVector3 lv = gBtObject->getLinearVelocity();
            ballVelocity = {lv.x(), lv.y(), lv.z()};
        }
        for (auto& obj : gSceneObjects) syncSceneObjectFromBullet(obj);
        return;
    }
    if (objectShape == 1) updatePhysicsBox(dt);
    else                  updatePhysicsSphere(dt);
}

// ── Gizmo / pick helpers ──────────────────────────────────────────────────────

static Vec3 getCameraEye() {
    float yr=yaw*pi/180.f, pr=pitch*pi/180.f;
    return {cameraDistance*std::cos(pr)*std::sin(yr),
            cameraDistance*std::sin(pr),
            cameraDistance*std::cos(pr)*std::cos(yr)};
}

static Mat4 getViewProjection() {
    Vec3 eye=getCameraEye();
    return multiply(
        perspective(45.f*pi/180.f,(float)framebufferWidth/framebufferHeight,0.1f,5000.f),
        lookAt(eye,{0,0,0},{0,1,0}));
}

// World position → window (screen) pixel.  Returns {-99999,…} if behind camera.
// Uses io.DisplaySize rather than glfwGetWindowSize(): on the web build the
// canvas is resized directly (ResizeCanvasForDPI), which never updates
// GLFW's own internally tracked window size, so glfwGetWindowSize() goes
// stale while io.DisplaySize (and thus ImGui's mouse hit-testing) stays
// correct. Using the same source as ImGui keeps screen-space math and
// on-screen rendering/clicking in agreement on both platforms.
static ImVec2 worldToScreen(Vec3 p, const Mat4& vp) {
    float x=vp.m[0]*p.x+vp.m[4]*p.y+vp.m[8]*p.z +vp.m[12];
    float y=vp.m[1]*p.x+vp.m[5]*p.y+vp.m[9]*p.z +vp.m[13];
    float w=vp.m[3]*p.x+vp.m[7]*p.y+vp.m[11]*p.z+vp.m[15];
    if (w<0.001f) return {-99999,-99999};
    ImVec2 disp = ImGui::GetIO().DisplaySize;
    return {(x/w+1.f)*0.5f*disp.x, (1.f-y/w)*0.5f*disp.y};
}

static float distToSeg2D(ImVec2 p, ImVec2 a, ImVec2 b) {
    float dx=b.x-a.x,dy=b.y-a.y,len2=dx*dx+dy*dy;
    if (len2<0.01f) { float ex=p.x-a.x,ey=p.y-a.y; return std::sqrt(ex*ex+ey*ey); }
    float t=std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/len2,0.f,1.f);
    float qx=a.x+t*dx,qy=a.y+t*dy;
    return std::sqrt((p.x-qx)*(p.x-qx)+(p.y-qy)*(p.y-qy));
}

struct CamRay { Vec3 origin, dir; };
static CamRay screenRay(double mx, double my) {
    ImVec2 disp = ImGui::GetIO().DisplaySize;
    float ndcX=(float)(2.0*mx/disp.x-1.0);
    float ndcY=(float)(1.0-2.0*my/disp.y);
    Vec3  eye=getCameraEye();
    Vec3  fwd=normalize({-eye.x,-eye.y,-eye.z});
    Vec3  right=normalize(cross(fwd,{0,1,0}));
    Vec3  up=cross(right,fwd);
    float aspect=disp.x/disp.y;
    float tanH=std::tan(45.f*pi/180.f*0.5f);
    return {eye, normalize(fwd+right*(ndcX*aspect*tanH)+up*(ndcY*tanH))};
}

// Ray-sphere pick against scene objects; returns nearest index or -1
static int pickSceneObject(double mx, double my) {
    CamRay r=screenRay(mx,my);
    float best=1e30f; int hit=-1;
    for (int i=0;i<(int)gSceneObjects.size();++i) {
        const SceneObject& o=gSceneObjects[i];
        float br=o.r*std::max({o.sx,o.sy,o.sz});
        Vec3 oc=o.pos-r.origin;
        float b=dot(oc,r.dir), disc=b*b-dot(oc,oc)+br*br;
        if (disc<0) continue;
        float t=b-std::sqrt(disc);
        if (t>0.01f&&t<best){best=t;hit=i;}
    }
    return hit;
}

// Quaternion → Euler degrees (pitch yaw roll) for initialising rotation drag
static Vec3 quatToEulerDeg(Quat q) {
    float sp=std::clamp(2.f*(q.w*q.y-q.z*q.x),-1.f,1.f);
    float pitch=std::asin(sp)*180.f/pi;
    float yawA =std::atan2(2.f*(q.w*q.z+q.x*q.y),1.f-2.f*(q.y*q.y+q.z*q.z))*180.f/pi;
    float roll =std::atan2(2.f*(q.w*q.x+q.y*q.z),1.f-2.f*(q.x*q.x+q.y*q.y))*180.f/pi;
    return {pitch,yawA,roll};
}

// Returns which gizmo element is under (mx,my): 0=X 1=Y 2=Z 3=center  -1=miss
static int pickGizmoAxis(double mx, double my) {
    if (gSelectedObjIdx<0||gSelectedObjIdx>=(int)gSceneObjects.size()) return -1;
    const SceneObject& obj=gSceneObjects[gSelectedObjIdx];
    Mat4 vp=getViewProjection();
    float gLen=cameraDistance*0.10f;
    ImVec2 mp{(float)mx,(float)my};
    ImVec2 cen=worldToScreen(obj.pos,vp);
    const Vec3 axes[3]={{1,0,0},{0,1,0},{0,0,1}};
    constexpr float THRESH=15.f;

    if (gGizmoMode==GizmoMode::Translate||gGizmoMode==GizmoMode::Scale) {
        float cd=std::sqrt((cen.x-mx)*(cen.x-mx)+(cen.y-my)*(cen.y-my));
        if (cd<THRESH) return 3;
        for (int ax=0;ax<3;++ax) {
            ImVec2 tip=worldToScreen(obj.pos+axes[ax]*gLen,vp);
            if (distToSeg2D(mp,cen,tip)<THRESH) return ax;
        }
    } else { // Rotate rings — sample a circle and find nearest point
        constexpr int S=48;
        for (int ax=0;ax<3;++ax) {
            float best=1e9f;
            for (int k=0;k<S;++k) {
                float a=2.f*pi*k/S;
                Vec3 pt;
                if(ax==0) pt={obj.pos.x,obj.pos.y+std::cos(a)*gLen,obj.pos.z+std::sin(a)*gLen};
                else if(ax==1) pt={obj.pos.x+std::cos(a)*gLen,obj.pos.y,obj.pos.z+std::sin(a)*gLen};
                else pt={obj.pos.x+std::cos(a)*gLen,obj.pos.y+std::sin(a)*gLen,obj.pos.z};
                ImVec2 s=worldToScreen(pt,vp);
                float d=std::sqrt((s.x-mx)*(s.x-mx)+(s.y-my)*(s.y-my));
                best=std::min(best,d);
            }
            if (best<THRESH) return ax;
        }
    }
    return -1;
}

static void startGizmoDrag(int axis, double mx, double my) {
    if (gSelectedObjIdx<0||gSelectedObjIdx>=(int)gSceneObjects.size()) return;
    SceneObject& obj=gSceneObjects[gSelectedObjIdx];
    syncSceneObjectFromBullet(obj);
    gGizmoDrag.active=true; gGizmoDrag.axis=axis;
    gGizmoDrag.startMX=mx; gGizmoDrag.startMY=my;
    gGizmoDrag.startPos=obj.pos;
    gGizmoDrag.startEuler=quatToEulerDeg(obj.orient);
    obj.euler=gGizmoDrag.startEuler;
    gGizmoDrag.startSx=obj.sx; gGizmoDrag.startSy=obj.sy; gGizmoDrag.startSz=obj.sz;
}

static void updateGizmoDrag(double mx, double my) {
    if (!gGizmoDrag.active||gSelectedObjIdx<0) return;
    SceneObject& obj=gSceneObjects[gSelectedObjIdx];
    float dx=(float)(mx-gGizmoDrag.startMX);
    float dy=(float)(my-gGizmoDrag.startMY);
    const Vec3 axes[3]={{1,0,0},{0,1,0},{0,0,1}};

    if (gGizmoMode==GizmoMode::Translate) {
        float gLen=cameraDistance*0.10f;
        if (gGizmoDrag.axis==3) {
            float spd=cameraDistance*0.002f;
            obj.pos={gGizmoDrag.startPos.x+dx*spd,
                     gGizmoDrag.startPos.y,
                     gGizmoDrag.startPos.z+dy*spd};
        } else {
            Mat4 vp=getViewProjection();
            ImVec2 sa=worldToScreen(gGizmoDrag.startPos,vp);
            ImVec2 sb=worldToScreen(gGizmoDrag.startPos+axes[gGizmoDrag.axis]*gLen,vp);
            float axDx=sb.x-sa.x,axDy=sb.y-sa.y,axLen2=axDx*axDx+axDy*axDy;
            if (axLen2>0.1f) {
                float proj=(dx*axDx+dy*axDy)/axLen2;
                obj.pos=gGizmoDrag.startPos+axes[gGizmoDrag.axis]*(proj*gLen);
            }
        }
        pushSceneObjectToBullet(obj);

    } else if (gGizmoMode==GizmoMode::Rotate) {
        Vec3 e=gGizmoDrag.startEuler;
        float deg=dx*0.5f;
        if      (gGizmoDrag.axis==0) e.x+=deg;
        else if (gGizmoDrag.axis==1) e.y+=deg;
        else if (gGizmoDrag.axis==2) e.z+=deg;
        else { e.x+=dx*0.3f; e.y+=dy*0.3f; }
        obj.euler=e;
        obj.orient=eulerToQuat(e.x,e.y,e.z);
        pushSceneObjectToBullet(obj);

    } else { // Scale
        float delta=dx*0.005f;
        float ns[3]={gGizmoDrag.startSx,gGizmoDrag.startSy,gGizmoDrag.startSz};
        if      (gGizmoDrag.axis==0) ns[0]=std::max(0.05f,ns[0]+delta);
        else if (gGizmoDrag.axis==1) ns[1]=std::max(0.05f,ns[1]+delta);
        else if (gGizmoDrag.axis==2) ns[2]=std::max(0.05f,ns[2]+delta);
        else { // uniform from center handle
            float u=std::max(0.05f,1.f+delta);
            ns[0]=std::max(0.05f,gGizmoDrag.startSx*u);
            ns[1]=std::max(0.05f,gGizmoDrag.startSy*u);
            ns[2]=std::max(0.05f,gGizmoDrag.startSz*u);
        }
        // Update the visual scale only; render() reads obj.sx/sy/sz directly
        // every frame, so this alone is enough for smooth live feedback.
        // Rebuilding the actual Bullet collision shape here would mean
        // deleting and reallocating a rigid body on every single mouse-move
        // event for the whole drag - rebuildSceneObjectShape() is deferred
        // to drag-release instead (see mouseButtonCallback).
        obj.sx=ns[0]; obj.sy=ns[1]; obj.sz=ns[2];
    }
}

// Rotation matrix: maps cone's local +Y to world axis ax (0=X 1=Y 2=Z)
static Mat4 rotYToAxis(int ax) {
    Mat4 m=identity();
    if (ax==0) {               // Rz(-90°): local Y → world X
        m.m[0]=0.f; m.m[1]=-1.f;
        m.m[4]=1.f; m.m[5]= 0.f;
    } else if (ax==2) {        // Rx(+90°): local Y → world Z
        m.m[5]=0.f;  m.m[6]=1.f;
        m.m[9]=-1.f; m.m[10]=0.f;
    }
    return m;
}

static void renderGizmos(const Mat4& vp) {
    if (gSelectedObjIdx<0||gSelectedObjIdx>=(int)gSceneObjects.size()) return;
    const SceneObject& obj=gSceneObjects[gSelectedObjIdx];
    float gLen=cameraDistance*0.10f;
    const Vec3 axes[3]={{1,0,0},{0,1,0},{0,0,1}};
    // X=red  Y=green  Z=blue
    const float CR[3]={1.f,0.18f,0.18f};
    const float CG[3]={0.18f,1.f,0.18f};
    const float CB[3]={0.18f,0.18f,1.f};

    // ── Line geometry (shafts or rings) ──────────────────────────────────────
    std::vector<Vertex> lines;
    auto addLine=[&](Vec3 a,Vec3 b,float r,float g,float bl){
        lines.push_back({a,{},r,g,bl,1.f});
        lines.push_back({b,{},r,g,bl,1.f});
    };

    if (gGizmoMode!=GizmoMode::Rotate) {
        for (int ax=0;ax<3;++ax) {
            bool act=gGizmoDrag.active&&gGizmoDrag.axis==ax;
            float br=act?1.3f:0.85f;
            addLine(obj.pos,obj.pos+axes[ax]*gLen,CR[ax]*br,CG[ax]*br,CB[ax]*br);
        }
    } else {
        constexpr int S=64;
        for (int ax=0;ax<3;++ax) {
            bool act=gGizmoDrag.active&&gGizmoDrag.axis==ax;
            float br=act?1.3f:0.85f;
            for (int k=0;k<S;++k) {
                float a0=2.f*pi*k/S, a1=2.f*pi*(k+1)/S;
                Vec3 p0,p1;
                if(ax==0){
                    p0={obj.pos.x,obj.pos.y+std::cos(a0)*gLen,obj.pos.z+std::sin(a0)*gLen};
                    p1={obj.pos.x,obj.pos.y+std::cos(a1)*gLen,obj.pos.z+std::sin(a1)*gLen};
                } else if(ax==1) {
                    p0={obj.pos.x+std::cos(a0)*gLen,obj.pos.y,obj.pos.z+std::sin(a0)*gLen};
                    p1={obj.pos.x+std::cos(a1)*gLen,obj.pos.y,obj.pos.z+std::sin(a1)*gLen};
                } else {
                    p0={obj.pos.x+std::cos(a0)*gLen,obj.pos.y+std::sin(a0)*gLen,obj.pos.z};
                    p1={obj.pos.x+std::cos(a1)*gLen,obj.pos.y+std::sin(a1)*gLen,obj.pos.z};
                }
                addLine(p0,p1,CR[ax]*br,CG[ax]*br,CB[ax]*br);
            }
        }
    }

    if (gGizmoLineMesh.vao==0) configureMesh(gGizmoLineMesh);
    gGizmoLineMesh.mode=GL_LINES;
    uploadVertices(gGizmoLineMesh,lines,GL_DYNAMIC_DRAW);

    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    glUniform3f(colorMultLocation,1.f,1.f,1.f);
    drawMesh(gGizmoLineMesh,vp,identity(),false);
    glLineWidth(1.f);

    // ── Arrowheads (translate) or cube handles (scale) ────────────────────────
    if (gGizmoMode!=GizmoMode::Rotate) {
        float headSz=gLen*0.13f;
        for (int ax=0;ax<3;++ax) {
            bool act=gGizmoDrag.active&&gGizmoDrag.axis==ax;
            float br=act?1.4f:1.f;
            glUniform3f(colorMultLocation,CR[ax]*br,CG[ax]*br,CB[ax]*br);

            Vec3 tip=obj.pos+axes[ax]*gLen;

            // Rotation: maps cone/box local Y to world axis
            Mat4 rot=rotYToAxis(ax);
            // Apply uniform scale into the rotation columns
            rot.m[0]*=headSz; rot.m[1]*=headSz; rot.m[2]*=headSz;
            rot.m[4]*=headSz; rot.m[5]*=headSz; rot.m[6]*=headSz;
            rot.m[8]*=headSz; rot.m[9]*=headSz; rot.m[10]*=headSz;

            if (gGizmoMode==GizmoMode::Translate) {
                // Cone: local y=-1..+1; place center at tip-axis*headSz
                Vec3 coneCenter=tip-axes[ax]*headSz;
                drawMesh(gConeMesh,vp,multiply(translation(coneCenter),rot),false);
            } else {
                // Small cube at tip
                Mat4 s=identity(); s.m[0]=s.m[5]=s.m[10]=headSz*0.65f;
                drawMesh(gUnitBoxMesh,vp,multiply(translation(tip),s),false);
            }
        }
        // Center handle: yellow cube (move/scale uniform)
        {
            bool act=gGizmoDrag.active&&gGizmoDrag.axis==3;
            float brt=act?1.f:0.85f;
            glUniform3f(colorMultLocation,brt,brt,0.f);
            float cs=gLen*0.10f;
            Mat4 s=identity(); s.m[0]=s.m[5]=s.m[10]=cs;
            drawMesh(gUnitBoxMesh,vp,multiply(translation(obj.pos),s),false);
        }
    }

    glEnable(GL_DEPTH_TEST);
    glUniform3f(colorMultLocation,1.f,1.f,1.f);
}

void render() {
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glClearColor(0.07f, 0.07f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    float yawRadians = yaw * pi / 180.0f;
    float pitchRadians = pitch * pi / 180.0f;
    Vec3 eye{cameraDistance * std::cos(pitchRadians) * std::sin(yawRadians),
             cameraDistance * std::sin(pitchRadians),
             cameraDistance * std::cos(pitchRadians) * std::cos(yawRadians)};
    Mat4 projection = perspective(45.0f * pi / 180.0f,
                                  static_cast<float>(framebufferWidth) / framebufferHeight,
                                  0.1f, 5000.0f);
    Mat4 viewProjection = multiply(projection, lookAt(eye, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    glUseProgram(shaderProgram);
    glUniform3f(eyeLocation, eye.x, eye.y, eye.z);
    // Fog: starts at camera distance, fully opaque at 4× camera distance.
    // Terrain/grid vertices are in local space; fog uses world-space distance from eye via vWorldPos.
    float fogStart = cameraDistance;
    float fogEnd   = cameraDistance * 4.0f;
    glUniform1f(fogStartLocation, fogStart);
    glUniform1f(fogEndLocation,   fogEnd);
    // Terrain and grid are in local space; translate to world by terrain center.
    Mat4 terrainModel = translation({gTerrainCX, 0.0f, gTerrainCZ});
    drawMesh(terrainMesh, viewProjection, terrainModel, true);
    drawMesh(gridMesh, viewProjection, terrainModel, false);
    glLineWidth(2.5f);
    drawMesh(axesMesh, viewProjection, identity(), false);
    glLineWidth(1.0f);
    // ── Scene objects ─────────────────────────────────────────────────────────
    for (int si = 0; si < (int)gSceneObjects.size(); ++si) {
        const SceneObject& obj = gSceneObjects[si];
        const ObjectMaterial& om = kObjectMaterials[obj.matIdx];
        float br = (si == gSelectedObjIdx) ? 1.4f : 1.0f; // highlight selected
        glUniform3f(colorMultLocation, om.r * br, om.g * br, om.b * br);

        // Build model: translate * rotate * non-uniform scale
        Mat4 rot = quatToMat4(obj.orient);
        float sx=obj.r*obj.sx, sy=obj.r*obj.sy, sz=obj.r*obj.sz;
        rot.m[0]*=sx; rot.m[1]*=sx; rot.m[2]*=sx;
        rot.m[4]*=sy; rot.m[5]*=sy; rot.m[6]*=sy;
        rot.m[8]*=sz; rot.m[9]*=sz; rot.m[10]*=sz;
        Mat4 mdl = multiply(translation(obj.pos), rot);

        Mesh* m = nullptr;
        switch (obj.shapeType) {
            case 0: m = &gUnitSphereMesh; break;
            case 1: m = &gUnitBoxMesh;    break;
            case 2: m = &gCylinderMesh;   break;
            case 3: m = &gConeMesh;       break;
            case 4: m = &gCapsuleMesh;    break;
            case 5: m = &gCarMesh;         break; // car
        }
        if (m) drawMesh(*m, viewProjection, mdl, true, si == gSelectedObjIdx);
    }
    glUniform3f(colorMultLocation, 1.0f, 1.0f, 1.0f); // reset

    // ── Transform gizmo overlay (drawn last, depth-test disabled internally) ──
    renderGizmos(viewProjection);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_P:
            gSimPaused = !gSimPaused;
            std::printf("Simulation %s\n", gSimPaused ? "PAUSED" : "RUNNING");
            break;
        case GLFW_KEY_PERIOD:   // '.' = single step while paused
            gSimStep = true;
            if (!gSimPaused) { gSimPaused = true; }
            break;
        case GLFW_KEY_C:
            currentColor = (currentColor + 1) % numColors;
            std::printf("Grid color: %s\n", gridColors[currentColor].name);
            createGridMesh();
            break;
        case GLFW_KEY_SPACE: autoRotate = !autoRotate; break;
        case GLFW_KEY_W: gGizmoMode = GizmoMode::Translate; break;
        case GLFW_KEY_E: gGizmoMode = GizmoMode::Rotate;    break;
        case GLFW_KEY_T: gGizmoMode = GizmoMode::Scale;     break;
        case GLFW_KEY_R: yaw = 35.0f; pitch = 25.0f; cameraDistance = 55.0f; break;
        case GLFW_KEY_LEFT: yaw -= 3.0f; break;
        case GLFW_KEY_RIGHT: yaw += 3.0f; break;
        case GLFW_KEY_UP: pitch -= 3.0f; break;
        case GLFW_KEY_DOWN: pitch += 3.0f; break;
        case GLFW_KEY_EQUAL: cameraDistance -= 1.0f; break;
        case GLFW_KEY_MINUS: cameraDistance += 1.0f; break;
        case GLFW_KEY_RIGHT_BRACKET:
            gridSize = std::min(gridSize + (gridSize < 100 ? 1 : 50), 2000);
            std::printf("Grid size: %d\n", gridSize);
            createTerrainMesh(); createGridMesh(); createAxesMesh();
            break;
        case GLFW_KEY_LEFT_BRACKET:
            gridSize = std::max(gridSize - (gridSize <= 100 ? 1 : 50), 2);
            std::printf("Grid size: %d\n", gridSize);
            createTerrainMesh(); createGridMesh(); createAxesMesh();
            break;
    }
    yaw = std::fmod(yaw + 360.0f, 360.0f);
    pitch = std::clamp(pitch, -89.0f, 89.0f);
    cameraDistance = std::clamp(cameraDistance, 3.0f, 2000.0f);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        // Use the position from the most recent cursorCallback rather than a
        // fresh glfwGetCursorPos(): on the web build, GLFW's own cursor-pos
        // tracking is in its stale internal window-size space (see
        // ApplyWebDisplayMetrics), whereas lastMouseX/Y is fed by the raw,
        // correctly-scaled coordinates from our own mouse/touch callbacks.
        double mx = lastMouseX, my = lastMouseY;

        // Priority: gizmo handle → object pick → camera orbit
        int gAxis = pickGizmoAxis(mx, my);
        if (gAxis >= 0) {
            startGizmoDrag(gAxis, mx, my);
        } else {
            int picked = pickSceneObject(mx, my);
            if (picked >= 0) {
                double now = glfwGetTime();
                bool isDouble = (picked == gLastClickedObj &&
                                 now - gLastClickTime < kDoubleClickSec);
                gSelectedObjIdx = picked;
                gGizmoMode      = isDouble ? GizmoMode::Rotate : GizmoMode::Translate;
                gLastClickTime  = now;
                gLastClickedObj = picked;
            } else {
                gSelectedObjIdx = -1;
                gLastClickedObj = -1;
                dragging = true;
            }
        }
    } else {                                // RELEASE
        // A scale drag only updates the visual sx/sy/sz each frame (see
        // updateGizmoDrag); catch the actual Bullet collision shape up to
        // the final size now that dragging has stopped.
        if (gGizmoDrag.active && gGizmoMode == GizmoMode::Scale &&
            gSelectedObjIdx >= 0 && gSelectedObjIdx < (int)gSceneObjects.size()) {
            rebuildSceneObjectShape(gSelectedObjIdx);
        }
        gGizmoDrag.active = false;
        dragging = false;
    }
}

void cursorCallback(GLFWwindow* window, double x, double y) {
    ImGui_ImplGlfw_CursorPosCallback(window, x, y);
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (gGizmoDrag.active) {
        updateGizmoDrag(x, y);
    } else if (dragging) {
        yaw   += static_cast<float>((x - lastMouseX) * 0.5);
        pitch += static_cast<float>((y - lastMouseY) * 0.5);
        yaw   = std::fmod(yaw + 360.0f, 360.0f);
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }
    lastMouseX = x;
    lastMouseY = y;
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xOffset, yOffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    cameraDistance = std::clamp(cameraDistance - static_cast<float>(yOffset), 3.0f, 2000.0f);
}

void framebufferSizeCallback(GLFWwindow*, int width, int height) {
    framebufferWidth = width;
    framebufferHeight = std::max(height, 1);
}

#if defined(__EMSCRIPTEN__)
// GLFW's Emscripten port sets the canvas's actual pixel buffer to exactly
// the width/height passed to glfwCreateWindow (900x700), ignoring both the
// canvas's real CSS display size and the display's pixel density — any
// earlier JS-side canvas sizing gets silently overwritten the moment the
// window is created. This resizes the real framebuffer to CSS size ×
// devicePixelRatio (verified via devtools: it was rendering at a flat
// 900x700 and being upscaled by the browser to fill the page, which is
// what caused the blur).
//
// Called every frame (see mainLoopIteration), not just once at startup +
// on resize events: this runs so early — the instant the wasm module
// finishes loading — that the browser may not have finished its first
// layout pass yet, so emscripten_get_element_css_size can read back 0 or a
// stale size at that exact moment, with no further resize event ever
// firing afterward to correct it since the window itself doesn't change
// size again. Re-checking every frame makes it self-correcting regardless
// of that startup race, at negligible cost since it's a no-op once the
// size actually matches.
void ResizeCanvasForDPI() {
    double cssWidth = 0.0, cssHeight = 0.0;
    emscripten_get_element_css_size("#canvas", &cssWidth, &cssHeight);
    double dpr = emscripten_get_device_pixel_ratio();
    int fbWidth  = std::max((int)std::round(cssWidth * dpr), 1);
    int fbHeight = std::max((int)std::round(cssHeight * dpr), 1);
    if (fbWidth == framebufferWidth && fbHeight == framebufferHeight) return;
    emscripten_set_canvas_element_size("#canvas", fbWidth, fbHeight);
    framebufferWidth  = fbWidth;
    framebufferHeight = fbHeight;
    glViewport(0, 0, fbWidth, fbHeight);
}
#endif

void printHelp() {
    std::printf("=== Physics Experiment ===\n");
    std::printf("  Left mouse drag : rotate camera\n");
    std::printf("  Arrow keys      : change view angle\n");
    std::printf("  Mouse wheel     : zoom in / out\n");
    std::printf("  c               : cycle grid color\n");
    std::printf("  Space           : toggle auto-rotation\n");
    std::printf("  r               : reset camera\n");
    std::printf("  ] / [           : grow / shrink space\n");
    std::printf("  Esc             : quit\n");
}

// ── Refined macOS glass UI — frosted panels, restrained accents ───────────────
static void setup3DStyle(float dpiScale) {
    ImGuiStyle& s = ImGui::GetStyle();

    s.WindowRounding    = 18.0f;
    s.ChildRounding     = 14.0f;
    s.FrameRounding     = 10.0f;
    s.PopupRounding     = 16.0f;
    s.ScrollbarRounding =  8.0f;
    s.GrabRounding      = 10.0f;
    s.TabRounding       = 10.0f;

    s.WindowBorderSize  = 1.0f;
    s.ChildBorderSize   = 1.0f;
    s.FrameBorderSize   = 0.0f;
    s.PopupBorderSize   = 1.0f;

    s.WindowPadding     = ImVec2(18.0f, 16.0f);
    s.FramePadding      = ImVec2(12.0f,  8.0f);
    s.CellPadding       = ImVec2(10.0f,  6.0f);
    s.ItemSpacing       = ImVec2(12.0f, 10.0f);
    s.ItemInnerSpacing  = ImVec2( 8.0f,  6.0f);
    s.IndentSpacing     = 22.0f;
    s.ScrollbarSize     =  6.0f;
    s.GrabMinSize       = 12.0f;
    s.WindowMinSize     = ImVec2(80.0f, 40.0f);
    s.WindowTitleAlign  = ImVec2(0.5f, 0.5f);
    s.ButtonTextAlign   = ImVec2(0.5f, 0.5f);

    if (dpiScale > 1.0f) s.ScaleAllSizes(dpiScale);

    ImVec4* c = s.Colors;

    c[ImGuiCol_Text]                  = ImVec4(0.96f, 0.97f, 0.99f, 1.00f);
    c[ImGuiCol_TextDisabled]          = ImVec4(0.54f, 0.56f, 0.62f, 1.00f);

    c[ImGuiCol_WindowBg]              = ImVec4(0.07f, 0.08f, 0.12f, 0.84f);
    c[ImGuiCol_ChildBg]               = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);
    c[ImGuiCol_PopupBg]               = ImVec4(0.08f, 0.09f, 0.14f, 0.94f);

    c[ImGuiCol_Border]                = ImVec4(1.00f, 1.00f, 1.00f, 0.16f);
    c[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    c[ImGuiCol_FrameBg]               = ImVec4(1.00f, 1.00f, 1.00f, 0.07f);
    c[ImGuiCol_FrameBgHovered]        = ImVec4(1.00f, 1.00f, 1.00f, 0.12f);
    c[ImGuiCol_FrameBgActive]         = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);

    c[ImGuiCol_TitleBg]               = ImVec4(0.06f, 0.07f, 0.11f, 0.00f);
    c[ImGuiCol_TitleBgActive]         = ImVec4(0.00f, 0.48f, 1.00f, 0.82f);
    c[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.06f, 0.07f, 0.11f, 0.58f);
    c[ImGuiCol_MenuBarBg]             = ImVec4(0.07f, 0.08f, 0.12f, 0.90f);

    c[ImGuiCol_ScrollbarBg]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_ScrollbarGrab]         = ImVec4(1.00f, 1.00f, 1.00f, 0.24f);
    c[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(1.00f, 1.00f, 1.00f, 0.38f);
    c[ImGuiCol_ScrollbarGrabActive]   = ImVec4(1.00f, 1.00f, 1.00f, 0.52f);

    c[ImGuiCol_CheckMark]             = ImVec4(0.00f, 0.48f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrab]            = ImVec4(1.00f, 1.00f, 1.00f, 0.88f);
    c[ImGuiCol_SliderGrabActive]      = ImVec4(0.00f, 0.48f, 1.00f, 1.00f);

    c[ImGuiCol_Button]                = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
    c[ImGuiCol_ButtonHovered]         = ImVec4(1.00f, 1.00f, 1.00f, 0.18f);
    c[ImGuiCol_ButtonActive]          = ImVec4(0.00f, 0.48f, 1.00f, 0.86f);

    c[ImGuiCol_Header]                = ImVec4(0.00f, 0.48f, 1.00f, 0.20f);
    c[ImGuiCol_HeaderHovered]         = ImVec4(0.00f, 0.48f, 1.00f, 0.32f);
    c[ImGuiCol_HeaderActive]          = ImVec4(0.00f, 0.48f, 1.00f, 0.48f);

    c[ImGuiCol_Separator]             = ImVec4(1.00f, 1.00f, 1.00f, 0.12f);
    c[ImGuiCol_SeparatorHovered]      = ImVec4(1.00f, 1.00f, 1.00f, 0.24f);
    c[ImGuiCol_SeparatorActive]       = ImVec4(0.00f, 0.48f, 1.00f, 0.72f);

    c[ImGuiCol_ResizeGrip]            = ImVec4(1.00f, 1.00f, 1.00f, 0.12f);
    c[ImGuiCol_ResizeGripHovered]     = ImVec4(1.00f, 1.00f, 1.00f, 0.24f);
    c[ImGuiCol_ResizeGripActive]      = ImVec4(0.00f, 0.48f, 1.00f, 0.72f);

    c[ImGuiCol_Tab]                   = ImVec4(1.00f, 1.00f, 1.00f, 0.04f);
    c[ImGuiCol_TabHovered]            = ImVec4(1.00f, 1.00f, 1.00f, 0.12f);
    c[ImGuiCol_TabActive]             = ImVec4(0.00f, 0.48f, 1.00f, 0.80f);
    c[ImGuiCol_TabUnfocused]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);
    c[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.00f, 0.48f, 1.00f, 0.44f);

    c[ImGuiCol_PlotLines]             = ImVec4(0.00f, 0.48f, 1.00f, 1.00f);
    c[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.62f, 0.00f, 1.00f);
    c[ImGuiCol_PlotHistogram]         = ImVec4(0.00f, 0.48f, 1.00f, 0.82f);
    c[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.62f, 0.00f, 1.00f);

    c[ImGuiCol_TextSelectedBg]        = ImVec4(0.00f, 0.48f, 1.00f, 0.30f);
    c[ImGuiCol_DragDropTarget]        = ImVec4(0.00f, 0.48f, 1.00f, 0.82f);
    c[ImGuiCol_NavHighlight]          = ImVec4(0.00f, 0.48f, 1.00f, 1.00f);
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.68f);
    c[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.00f, 0.00f, 0.00f, 0.38f);
    c[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
}

// ── Draw a shape icon onto ImDrawList at position p in a sz×sz box ────────────
static void DrawLibIcon(ImDrawList* dl, ImVec2 p, float sz, int shape, ImU32 fill, ImU32 edge) {
    float r  = sz * 0.34f;
    float cx = p.x + sz*0.5f, cy = p.y + sz*0.50f;
    switch (shape) {
        case 0: // sphere
            dl->AddCircleFilled({cx,cy},r,fill,28);
            dl->AddCircle({cx,cy},r,edge,28,1.5f);
            break;
        case 1: // box
            dl->AddRectFilled({cx-r,cy-r},{cx+r,cy+r},fill);
            dl->AddRect({cx-r,cy-r},{cx+r,cy+r},edge,0.f,0,1.5f);
            break;
        case 2: { // cylinder: body rect + top face suggestion
            float wr=r*0.68f, hr=r;
            dl->AddRectFilled({cx-wr,cy-hr},{cx+wr,cy+hr},fill);
            dl->AddRectFilled({cx-wr,cy-hr},{cx+wr,cy-hr+r*0.35f},IM_COL32(255,255,255,55));
            dl->AddRect({cx-wr,cy-hr},{cx+wr,cy+hr},edge,0.f,0,1.5f);
            break;
        }
        case 3: // cone
            dl->AddTriangleFilled({cx,cy-r},{cx-r,cy+r*0.8f},{cx+r,cy+r*0.8f},fill);
            dl->AddTriangle({cx,cy-r},{cx-r,cy+r*0.8f},{cx+r,cy+r*0.8f},edge,1.5f);
            break;
        case 4: { // capsule: pill shape
            float wr=r*0.52f, hr=r*0.48f;
            dl->AddRectFilled({cx-wr,cy-hr},{cx+wr,cy+hr},fill);
            dl->AddCircleFilled({cx,cy-hr},wr,fill,14);
            dl->AddCircleFilled({cx,cy+hr},wr,fill,14);
            break;
        }
        case 5: { // car: side-profile silhouette (body + cabin + wheels)
            float bw = sz*0.42f, bh = sz*0.17f;
            float cw = sz*0.27f, ch = sz*0.13f;
            float wr = sz*0.10f;
            float gy = cy + sz*0.16f; // ground reference
            // body
            dl->AddRectFilled({cx-bw,gy-bh},{cx+bw,gy},fill);
            // cabin/roof
            dl->AddRectFilled({cx-cw,gy-bh-ch},{cx+cw,gy-bh},fill);
            // windshield highlight
            dl->AddRectFilled({cx-cw+3,gy-bh-ch+3},{cx+cw-3,gy-bh-2},IM_COL32(180,220,255,60));
            // outline
            dl->AddRect({cx-bw,gy-bh},{cx+bw,gy},edge,0.f,0,1.5f);
            dl->AddRect({cx-cw,gy-bh-ch},{cx+cw,gy-bh},edge,0.f,0,1.2f);
            // wheels (dark)
            ImU32 wclr = IM_COL32(35,35,35,255);
            dl->AddCircleFilled({cx+bw*0.58f,gy+wr*0.35f},wr,wclr,20);
            dl->AddCircle({cx+bw*0.58f,gy+wr*0.35f},wr,edge,20,1.0f);
            dl->AddCircleFilled({cx-bw*0.58f,gy+wr*0.35f},wr,wclr,20);
            dl->AddCircle({cx-bw*0.58f,gy+wr*0.35f},wr,edge,20,1.0f);
            break;
        }
    }
}

// ── Export helpers ─────────────────────────────────────────────────────────────
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
// Files written via std::ofstream land in Emscripten's in-memory MEMFS, which
// isn't visible to the visitor on its own — this reads it back out and hands
// it to the browser as a real download, same UX as the native app writing to
// disk directly.
EM_JS(void, downloadFileJS, (const char* path, const char* mime), {
    const p = UTF8ToString(path);
    const data = FS.readFile(p);
    const blob = new Blob([data], { type: UTF8ToString(mime) });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = p;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});
#endif

static std::string makeTimestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

static void exportTimeSeriesCSV() {
    if (gTimeSeries.empty()) return;
    std::string path = "experiment_" + makeTimestamp() + ".csv";
    std::ofstream f(path);
    f << "time_s,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,speed_mps,ke_J,altitude_m\n";
    for (auto& fr : gTimeSeries)
        f << fr.t << "," << fr.pos.x << "," << fr.pos.y << "," << fr.pos.z << ","
          << fr.vel.x << "," << fr.vel.y << "," << fr.vel.z << ","
          << fr.speed << "," << fr.ke << "," << fr.altitude << "\n";
    f.close();
    std::printf("Exported %d frames → %s\n", (int)gTimeSeries.size(), path.c_str());
#if defined(__EMSCRIPTEN__)
    downloadFileJS(path.c_str(), "text/csv");
#endif
}

static void exportExperimentJSON() {
    std::string path = "experiment_" + makeTimestamp() + ".json";
    std::ofstream f(path);
    const ObjectMaterial& om = kObjectMaterials[objectMaterialIdx];
    const GroundMaterial& gm = kGroundMaterials[groundMaterialIdx];
    f << "{\n";
    f << "  \"experiment\": {\n";
    f << "    \"object_material\": \"" << om.name << "\",\n";
    f << "    \"ground_material\": \"" << gm.name << "\",\n";
    f << "    \"gravity_env\": \"" << kGravities[gravityEnvIdx].name << "\",\n";
    f << "    \"gravity_mps2\": " << kGravities[gravityEnvIdx].g << ",\n";
    f << "    \"ball_radius_m\": " << ballRadius << ",\n";
    f << "    \"ball_mass_kg\": " << computeMass() << ",\n";
    f << "    \"drop_height_m\": " << ballDropHeight << ",\n";
    f << "    \"impact_velocity_mps\": " << std::sqrt(2.0f*kGravities[gravityEnvIdx].g*ballDropHeight) << ",\n";
    if (gImpactSnap.valid) {
        f << "    \"measured_impact\": {\n";
        f << "      \"time_s\": " << gImpactSnap.time << ",\n";
        f << "      \"speed_mps\": " << gImpactSnap.speed << ",\n";
        f << "      \"impulse_Ns\": " << gImpactSnap.impulse << ",\n";
        f << "      \"crater_count\": " << gImpactSnap.craterCount << "\n";
        f << "    },\n";
    }
    f << "    \"timeseries_frames\": " << (int)gTimeSeries.size() << "\n";
    f << "  }\n}\n";
    f.close();
    std::printf("Exported settings → %s\n", path.c_str());
#if defined(__EMSCRIPTEN__)
    downloadFileJS(path.c_str(), "application/json");
#endif
}

// ── 5 macOS system accent themes — shared glass base, restrained tint ─────────
// Fields: wbg  tbg  fbg  fbh  sg   cm   btn  bth  bta  hdr  hdh  tab  tba  tbh  sep
struct ThC {
    ImVec4 wbg,tbg,fbg,fbh,sg,cm,btn,bth,bta,hdr,hdh,tab,tba,tbh,sep;
};
static const ThC kThC[5] = {
  // 0 Midnight — system blue
  {{.07f,.08f,.12f,.84f},{.00f,.48f,1.0f,.82f},{1,1,1,.07f},{1,1,1,.12f},
   {1,1,1,.88f},{.00f,.48f,1,1},{1,1,1,.10f},{1,1,1,.18f},
   {.00f,.48f,1,.86f},{.00f,.48f,1,.18f},{.00f,.48f,1,.30f},
   {1,1,1,.04f},{.00f,.48f,1,.80f},{1,1,1,.12f},{1,1,1,.12f}},
  // 1 Solar — system orange
  {{.07f,.08f,.12f,.84f},{1.0f,.58f,.00f,.82f},{1,1,1,.07f},{1,1,1,.12f},
   {1,1,1,.88f},{1,.58f,.00f,1},{1,1,1,.10f},{1,1,1,.18f},
   {1,.58f,.00f,.86f},{1,.58f,.00f,.18f},{1,.58f,.00f,.30f},
   {1,1,1,.04f},{1,.58f,.00f,.80f},{1,1,1,.12f},{1,1,1,.12f}},
  // 2 Mint — system teal
  {{.07f,.08f,.12f,.84f},{.00f,.78f,.74f,.82f},{1,1,1,.07f},{1,1,1,.12f},
   {1,1,1,.88f},{.00f,.78f,.74f,1},{1,1,1,.10f},{1,1,1,.18f},
   {.00f,.78f,.74f,.86f},{.00f,.78f,.74f,.18f},{.00f,.78f,.74f,.30f},
   {1,1,1,.04f},{.00f,.78f,.74f,.80f},{1,1,1,.12f},{1,1,1,.12f}},
  // 3 Bloom — system purple
  {{.07f,.08f,.12f,.84f},{.68f,.32f,.87f,.82f},{1,1,1,.07f},{1,1,1,.12f},
   {1,1,1,.88f},{.68f,.32f,.87f,1},{1,1,1,.10f},{1,1,1,.18f},
   {.68f,.32f,.87f,.86f},{.68f,.32f,.87f,.18f},{.68f,.32f,.87f,.30f},
   {1,1,1,.04f},{.68f,.32f,.87f,.80f},{1,1,1,.12f},{1,1,1,.12f}},
  // 4 Rose — system pink
  {{.07f,.08f,.12f,.84f},{1.0f,.18f,.33f,.82f},{1,1,1,.07f},{1,1,1,.12f},
   {1,1,1,.88f},{1,.18f,.33f,1},{1,1,1,.10f},{1,1,1,.18f},
   {1,.18f,.33f,.86f},{1,.18f,.33f,.18f},{1,.18f,.33f,.30f},
   {1,1,1,.04f},{1,.18f,.33f,.80f},{1,1,1,.12f},{1,1,1,.12f}},
};
static void pushThemeColors(int t) {
    const ThC& c = kThC[t >= 0 && t < 5 ? t : 0];
    ImGui::PushStyleColor(ImGuiCol_WindowBg,       c.wbg);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive,  c.tbg);
    ImGui::PushStyleColor(ImGuiCol_FrameBg,        c.fbg);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, c.fbh);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab,     c.sg);
    ImGui::PushStyleColor(ImGuiCol_CheckMark,      c.cm);
    ImGui::PushStyleColor(ImGuiCol_Button,         c.btn);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  c.bth);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,   c.bta);
    ImGui::PushStyleColor(ImGuiCol_Header,         c.hdr);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,  c.hdh);
    ImGui::PushStyleColor(ImGuiCol_Tab,            c.tab);
    ImGui::PushStyleColor(ImGuiCol_TabActive,      c.tba);
    ImGui::PushStyleColor(ImGuiCol_TabHovered,     c.tbh);
    ImGui::PushStyleColor(ImGuiCol_Separator,      c.sep);
}

static ImVec4 themeAccent(int t = gTheme) {
    const ThC& c = kThC[t >= 0 && t < 5 ? t : 0];
    return c.cm;
}

static ImU32 toU32(ImVec4 c) {
    return ImGui::ColorConvertFloat4ToU32(c);
}

static ImVec4 alphaOf(ImVec4 c, float alpha) {
    c.w = alpha;
    return c;
}

static bool macSegmentButton(const char* label, bool selected, ImVec2 size) {
    ImVec4 accent = themeAccent();
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_Button,
        selected ? alphaOf(accent, 0.86f) : ImVec4(1,1,1,0.08f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
        selected ? alphaOf(accent, 0.94f) : ImVec4(1,1,1,0.14f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, alphaOf(accent, 0.90f));
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
    return clicked;
}

static void drawStatusPill(ImDrawList* dl, ImVec2 pos, const char* text, ImVec4 color) {
    ImVec2 ts = ImGui::CalcTextSize(text);
    ImVec2 min = pos;
    ImVec2 max = {pos.x + ts.x + 16.0f, pos.y + 22.0f};
    dl->AddRectFilled(min, max, toU32(alphaOf(color, 0.12f)), 10.0f);
    dl->AddRect(min, max, toU32(alphaOf(color, 0.22f)), 10.0f, 0, 0.8f);
    dl->AddCircleFilled({pos.x + 8.0f, pos.y + 11.0f}, 2.5f, toU32(color), 12);
    dl->AddText({pos.x + 14.0f, pos.y + 4.0f}, IM_COL32(220,224,232,220), text);
}

static void drawMacPanelSurface() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetWindowPos();
    ImVec2 s = ImGui::GetWindowSize();
    ImVec2 q = {p.x + s.x, p.y + s.y};
    float r = ImGui::GetStyle().WindowRounding;
    ImVec4 accent = themeAccent();

    dl->AddRectFilled({p.x + 1.0f, p.y + 6.0f}, {q.x + 1.0f, q.y + 6.0f},
        IM_COL32(0,0,0,28), r);
    dl->AddRectFilled({p.x + 2.0f, p.y + 3.0f}, {q.x + 2.0f, q.y + 3.0f},
        IM_COL32(0,0,0,18), r);

    dl->AddRectFilled(p, q, IM_COL32(18,20,26,214), r);
    dl->AddRectFilled(p, {q.x, p.y + 64.0f}, IM_COL32(24,26,34,208), r);

    dl->AddLine({p.x + r, p.y + 64.0f}, {q.x - r, p.y + 64.0f},
        IM_COL32(255,255,255,18), 1.0f);
    dl->AddLine({p.x + r, p.y + 1.0f}, {q.x - r, p.y + 1.0f},
        toU32(alphaOf(accent, 0.55f)), 1.0f);
    dl->AddRect(p, q, IM_COL32(255,255,255,22), r, 0, 1.0f);
}

static void drawMainPanelHeader() {
    drawMacPanelSurface();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetWindowPos();
    ImVec2 s = ImGui::GetWindowSize();

    dl->AddText({p.x + 16.0f, p.y + 14.0f}, IM_COL32(245,247,252,255),
        T("Physics Experiment", "物理実験"));
    dl->AddText({p.x + 16.0f, p.y + 34.0f}, IM_COL32(148,154,168,220),
        T("3D impact laboratory", "3D 衝突実験ラボ"));

    char fpsText[32];
    std::snprintf(fpsText, sizeof(fpsText), "FPS %.0f", ImGui::GetIO().Framerate);
    char objText[32];
    std::snprintf(objText, sizeof(objText), "%d objects", (int)gSceneObjects.size());

    drawStatusPill(dl, {p.x + s.x - 220.0f, p.y + 72.0f},
        gSimPaused ? T("Paused", "一時停止") : T("Running", "実行中"),
        gSimPaused ? ImVec4(0.95f,0.62f,0.28f,1) : ImVec4(0.34f,0.78f,0.52f,1));
    drawStatusPill(dl, {p.x + s.x - 132.0f, p.y + 72.0f}, fpsText,
        ImVec4(0.62f,0.68f,0.78f,1));
    drawStatusPill(dl, {p.x + 16.0f, p.y + 72.0f}, objText,
        ImVec4(0.62f,0.68f,0.78f,1));

    float segW = 148.0f;
    ImGui::SetCursorScreenPos({p.x + s.x - segW - 16.0f, p.y + 18.0f});
    bool jaActive = (gLang == Lang::JA);
    float half = (segW - 4.0f) * 0.5f;
    bool jpClicked = macSegmentButton("JP", jaActive, {half, 28.0f});
    ImGui::SameLine(0.0f, 4.0f);
    bool enClicked = macSegmentButton("EN", !jaActive, {half, 28.0f});
    if (jpClicked && !jaActive) gLang = Lang::JA;
    if (enClicked && jaActive) gLang = Lang::EN;

    ImGui::SetCursorScreenPos({p.x + 16.0f, p.y + 100.0f});
}

#if defined(__EMSCRIPTEN__)
// Dear ImGui's standard high-DPI model: DisplaySize in CSS/logical units,
// DisplayFramebufferScale carrying the device pixel ratio. Must run after
// ImGui_ImplGlfw_NewFrame(), since that's what writes io.DisplaySize in the
// first place.
static void ApplyWebDisplayMetrics() {
    double cssWidth = 0, cssHeight = 0;
    int canvasPixelWidth = 0, canvasPixelHeight = 0;
    emscripten_get_element_css_size("#canvas", &cssWidth, &cssHeight);
    emscripten_get_canvas_element_size("#canvas", &canvasPixelWidth, &canvasPixelHeight);

    if (cssWidth <= 0.0 || cssHeight <= 0.0) return;

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)cssWidth, (float)cssHeight);
    io.DisplayFramebufferScale = ImVec2(
        (float)(canvasPixelWidth  / cssWidth),
        (float)(canvasPixelHeight / cssHeight));
}
#endif

void renderHUD() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
#if defined(__EMSCRIPTEN__)
    ApplyWebDisplayMetrics();
#endif
    ImGui::NewFrame();

    // Refined glass rounding per frame
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,    10.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,   18.f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding,     10.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding,    14.f);
    pushThemeColors(gTheme);

    // ── Per-object context toolbar (appears beside selected object) ─────────
    if (gSelectedObjIdx >= 0 && gSelectedObjIdx < (int)gSceneObjects.size()) {
        const SceneObject& selObj = gSceneObjects[gSelectedObjIdx];
        Mat4 vp = getViewProjection();
        ImVec2 sp = worldToScreen(selObj.pos, vp);

        if (sp.x > -9000.f) {
            constexpr float IBW = 66.f, BH = 54.f, GAP = 6.f, PAD = 8.f;
            constexpr float TBW = IBW*3.f + GAP*2.f + PAD*2.f;
            constexpr float TBH = BH + 10.f + 30.f + PAD*2.f; // estimated toolbar height

            // World-space bounding radius → screen-space radius
            float worldR = selObj.r * std::max({selObj.sx, selObj.sy, selObj.sz});
            ImVec2 spEdge = worldToScreen(
                {selObj.pos.x + worldR, selObj.pos.y, selObj.pos.z}, vp);
            float screenR = (spEdge.x > -9000.f)
                ? std::abs(spEdge.x - sp.x) : worldR * 40.f;
            screenR = std::max(screenR, 20.f);

            ImVec2 disp = ImGui::GetIO().DisplaySize;
            constexpr float MARGIN = 14.f;
            // Place toolbar centred above the object; fall back to below if near top
            float tx = sp.x - TBW * 0.5f;
            float ty = sp.y - screenR - TBH - MARGIN;
            if (ty < 4.f) ty = sp.y + screenR + MARGIN;
            tx = std::max(tx, 4.f);
            tx = std::min(tx, disp.x - TBW - 4.f);
            ty = std::max(ty, 4.f);
            ty = std::min(ty, disp.y - TBH - 4.f);

            ImGui::SetNextWindowPos(ImVec2(tx, ty), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(TBW, 0.f), ImGuiCond_Always);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f,0.08f,0.12f,0.88f));
            ImGui::PushStyleColor(ImGuiCol_Border,   ImVec4(1.00f,1.00f,1.00f,0.16f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(PAD, PAD));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,   ImVec2(GAP, GAP));
            ImGui::Begin("##objtoolbar", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoFocusOnAppearing);

            struct ModeInfo {
                GizmoMode mode; ImVec4 colOn;
                const char* tip;   const char* tipJA;
                const char* label; const char* labelJA;
            };
            static const ModeInfo modes[] = {
                { GizmoMode::Translate, {0.18f,0.45f,0.88f,1.f}, "Move [W]",   "移動 [W]",    "Move",   "移動"   },
                { GizmoMode::Rotate,    {0.10f,0.58f,0.20f,1.f}, "Rotate [E]", "回転 [E]",    "Rotate", "回転"   },
                { GizmoMode::Scale,     {0.78f,0.45f,0.08f,1.f}, "Scale [T]",  "拡大縮小[T]", "Scale",  "拡大縮小"},
            };
            constexpr float LABEL_H = 18.f;

            for (int i = 0; i < 3; ++i) {
                bool sel = (gGizmoMode == modes[i].mode);
                ImVec4 colOn  = modes[i].colOn;
                ImVec4 colOff = {0.18f,0.19f,0.25f,0.90f};
                ImGui::PushStyleColor(ImGuiCol_Button,
                    sel ? ImVec4(colOn.x,colOn.y,colOn.z,0.30f) : colOff);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                    sel ? ImVec4(colOn.x,colOn.y,colOn.z,0.42f)
                        : ImVec4(0.32f,0.33f,0.39f,1.f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                    ImVec4(colOn.x,colOn.y,colOn.z,0.55f));
                ImVec2 bpos = ImGui::GetCursorScreenPos();
                ImGui::PushID(i);
                if (ImGui::Button("##gm", ImVec2(IBW, BH))) gGizmoMode = modes[i].mode;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", (gLang==Lang::JA) ? modes[i].tipJA : modes[i].tip);
                ImGui::PopID();
                ImGui::PopStyleColor(3);

                ImDrawList* dl = ImGui::GetWindowDrawList();
                float r = ImGui::GetStyle().FrameRounding;
                if (sel) {
                    // Bright top accent bar makes the active mode obvious at a glance,
                    // without needing to compare subtle fill-color differences.
                    dl->AddRectFilled({bpos.x, bpos.y}, {bpos.x + IBW, bpos.y + 3.f},
                        toU32(colOn), r);
                    dl->AddRect({bpos.x, bpos.y}, {bpos.x + IBW, bpos.y + BH},
                        toU32(alphaOf(colOn, 0.9f)), r, 0, 1.5f);
                }

                ImVec2 bc = {bpos.x + IBW*0.5f, bpos.y + (BH - LABEL_H)*0.5f};
                ImU32 ic = IM_COL32(255,255,255, sel ? 255 : 190);
                constexpr float S = 9.5f, A = 4.6f;

                if (i == 0) {
                    dl->AddLine({bc.x,bc.y},{bc.x+S,bc.y},ic,1.5f);
                    dl->AddTriangleFilled({bc.x+S+A,bc.y},{bc.x+S-1,bc.y-A*.55f},{bc.x+S-1,bc.y+A*.55f},ic);
                    dl->AddLine({bc.x,bc.y},{bc.x-S,bc.y},ic,1.5f);
                    dl->AddTriangleFilled({bc.x-S-A,bc.y},{bc.x-S+1,bc.y-A*.55f},{bc.x-S+1,bc.y+A*.55f},ic);
                    dl->AddLine({bc.x,bc.y},{bc.x,bc.y-S},ic,1.5f);
                    dl->AddTriangleFilled({bc.x,bc.y-S-A},{bc.x-A*.55f,bc.y-S+1},{bc.x+A*.55f,bc.y-S+1},ic);
                    dl->AddLine({bc.x,bc.y},{bc.x,bc.y+S},ic,1.5f);
                    dl->AddTriangleFilled({bc.x,bc.y+S+A},{bc.x-A*.55f,bc.y+S-1},{bc.x+A*.55f,bc.y+S-1},ic);
                    dl->AddCircleFilled(bc,2.f,ic);
                } else if (i == 1) {
                    float ar = S * 0.88f;
                    float sa = -2.8f, ea = 0.65f;
                    dl->PathArcTo(bc, ar, sa, ea, 24);
                    dl->PathStroke(ic, false, 2.0f);
                    float cosE=cosf(ea), sinE=sinf(ea);
                    float cosP=cosf(ea-1.5708f), sinP=sinf(ea-1.5708f);
                    ImVec2 tip={bc.x+cosE*(ar+A*.9f), bc.y+sinE*(ar+A*.9f)};
                    ImVec2 s1 ={bc.x+cosP*A*.5f+cosE*ar, bc.y+sinP*A*.5f+sinE*ar};
                    ImVec2 s2 ={bc.x-cosP*A*.5f+cosE*ar, bc.y-sinP*A*.5f+sinE*ar};
                    dl->AddTriangleFilled(tip,s1,s2,ic);
                } else {
                    dl->AddLine({bc.x-S,bc.y},{bc.x+S,bc.y},ic,1.5f);
                    dl->AddTriangleFilled({bc.x-S-A,bc.y},{bc.x-S+1,bc.y-A*.55f},{bc.x-S+1,bc.y+A*.55f},ic);
                    dl->AddTriangleFilled({bc.x+S+A,bc.y},{bc.x+S-1,bc.y-A*.55f},{bc.x+S-1,bc.y+A*.55f},ic);
                    dl->AddLine({bc.x,bc.y-5.5f},{bc.x,bc.y+5.5f},ic,1.8f);
                }

                const char* label = (gLang==Lang::JA) ? modes[i].labelJA : modes[i].label;
                ImVec2 ts = ImGui::CalcTextSize(label);
                dl->AddText({bpos.x + (IBW - ts.x)*0.5f, bpos.y + BH - LABEL_H - 2.f},
                    IM_COL32(255,255,255, sel ? 235 : 165), label);

                if (i < 2) ImGui::SameLine(0.f, GAP);
            }

            ImGui::Dummy(ImVec2(0.f, 6.f));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.f, 2.f));

            ImGui::TextColored(alphaOf(themeAccent(), 0.92f),
                "\xe2\x97\x86 %s", selObj.label);
            ImGui::SameLine(TBW - PAD*2.f - 84.f);
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.86f,0.20f,0.17f,0.85f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.96f,0.30f,0.26f,0.95f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.76f,0.15f,0.13f,1.00f));
            if (ImGui::Button(T("Delete","削除"), ImVec2(84.f, 26.f)))
                removeSceneObject(gSelectedObjIdx);
            ImGui::PopStyleColor(3);

            ImGui::End();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(2);
        }
    }

    // ── Drag-drop overlay (rendered first so combined panel sits on top) ────────
    if (ImGui::GetDragDropPayload() != nullptr) {
        ImVec2 dropDisp = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(0.f, 0.f));
        ImGui::SetNextWindowSize(dropDisp);
        ImGui::SetNextWindowBgAlpha(0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
        ImGui::Begin("##libdrop", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        ImGui::InvisibleButton("##vpdrop", dropDisp);
        if (ImGui::BeginDragDropTarget()) {
            ImGui::GetForegroundDrawList()->AddRect(
                ImVec2(4.f,4.f),
                ImVec2(dropDisp.x-4.f, dropDisp.y-4.f),
                toU32(alphaOf(themeAccent(), 0.55f)),0.f,0,2.f);
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("LIB_OBJECT")) {
                const LibDragPayload* ld = static_cast<const LibDragPayload*>(p->Data);
                if (ld->entryIdx>=0 && ld->entryIdx<kNumLibEntries) {
                    const LibEntry& le = kLibEntries[ld->entryIdx];
                    int vi = std::min(ld->variantIdx, le.nVariants-1);
                    const LibVariant& lv = le.variants[vi];
                    int mat = (lv.matHint>=0) ? lv.matHint : gLibMat;
                    addSceneObject(le.baseShape, mat, gLibSize, lv.sx, lv.sy, lv.sz);
                }
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::End();
    }

    // ── Panel state ───────────────────────────────────────────────────────────
    static bool  gPanelOpen    = true;
    static float gStoredPanelW = 445.f;

    // ── Toggle button — floats at top-right outside the box ──────────────────
    {
        constexpr float BZ = 26.f;
        float bx = 12.f + (gPanelOpen ? gStoredPanelW + 6.f : 0.f);
        ImGui::SetNextWindowPos(ImVec2(bx, 12.f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(BZ, BZ), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,    ImVec2(0,0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::Begin("##paneltog", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::PopStyleVar(2);
        ImVec2 tp = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##t", ImVec2(BZ, BZ));
        bool hov = ImGui::IsItemHovered();
        if (ImGui::IsItemClicked()) gPanelOpen = !gPanelOpen;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImU32 bg  = hov ? IM_COL32(255,255,255,42) : IM_COL32(255,255,255,18);
        ImU32 brd = hov ? IM_COL32(255,255,255,72) : IM_COL32(255,255,255,34);
        dl->AddRectFilled(tp, {tp.x+BZ,tp.y+BZ}, bg,  BZ*0.5f);
        dl->AddRect      (tp, {tp.x+BZ,tp.y+BZ}, brd, BZ*0.5f, 0, 0.8f);
        float cx = tp.x+BZ*0.5f, cy = tp.y+BZ*0.5f;
        ImU32 ac = IM_COL32(255,255,255, hov?255:200);
        if (gPanelOpen) // ◀ collapse
            dl->AddTriangleFilled({cx-5.f,cy},{cx+4.f,cy-5.5f},{cx+4.f,cy+5.5f},ac);
        else            // ▶ expand
            dl->AddTriangleFilled({cx+5.f,cy},{cx-4.f,cy-5.5f},{cx-4.f,cy+5.5f},ac);
        ImGui::End();
    }

    // ── Main panel (only when open) ───────────────────────────────────────────
    if (gPanelOpen) {
    // Sized from io.DisplaySize (CSS pixels), not framebufferHeight (physical
    // canvas pixels) — on any Retina/high-DPI display those differ by the
    // device pixel ratio, and using the physical value here made the window
    // roughly twice as tall as the visible viewport, with no way to reach
    // the part hanging off the bottom (NoScrollbar/NoScrollWithMouse below
    // also blocked panning to it even when the window height was otherwise
    // fine, e.g. on a short browser window).
    ImVec2 disp = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(12.f, 12.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(gStoredPanelW, disp.y - 24.f), ImGuiCond_Always);
    ImGui::Begin("##mainpanel", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize);
    gStoredPanelW = ImGui::GetWindowSize().x;

    drawMainPanelHeader();

    if (ImGui::BeginTabBar("##maintabs")) {
    if (ImGui::BeginTabItem(T("Environment", "環境"))) {
    ImGui::BeginChild("##envscroll", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);

    // ── Ground ───────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader(T("Ground", "地面"), ImGuiTreeNodeFlags_DefaultOpen)) {

    ImGui::Text("%s", T("Shape", "形状"));
    const char* groundNames[] = {
        T("Flat Plane",    "平坦面"),
        T("Bumpy Math",    "凹凸数学面"),
        T("Rolling Hills", "丘陵地形"),
        T("Bowl",          "凹形曲面"),
    };
    if (ImGui::Combo("##gshape", &groundType, groundNames, 4)) {
        createTerrainMesh(); createGridMesh();
        buildBulletWorld();
        resetBall();
    }

    ImGui::Text("%s", T("Surface", "表面材質"));
    constexpr int nGnd = sizeof(kGroundMaterials)/sizeof(kGroundMaterials[0]);
    const char* gndNames[nGnd];
    for (int i=0;i<nGnd;i++)
        gndNames[i] = (gLang == Lang::JA) ? kGroundMatNamesJA[i] : kGroundMaterials[i].name;
    if (ImGui::Combo("##gmat", &groundMaterialIdx, gndNames, nGnd)) {
        buildBulletWorld();
        resetBall();
    }
    } // CollapsingHeader Ground

    // ── Environment ──────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader(T("Environment & Forces", "環境と力"), ImGuiTreeNodeFlags_DefaultOpen)) {

    ImGui::Text("%s", T("Gravity", "重力加速度"));
    constexpr int nGrav = sizeof(kGravities)/sizeof(kGravities[0]);
    const char* gravNames[nGrav];
    for (int i=0;i<nGrav;i++)
        gravNames[i] = (gLang == Lang::JA) ? kGravNamesJA[i] : kGravities[i].name;
    if (ImGui::Combo("##grav", &gravityEnvIdx, gravNames, nGrav)) {
        if (gBtWorld)
            gBtWorld->setGravity(btVector3(0,-kGravities[gravityEnvIdx].g,0));
    }
    ImGui::TextColored(ImVec4(0.58f,0.72f,0.88f,1.f), "  g = %.2f m/s²",
                       kGravities[gravityEnvIdx].g);

    ImGui::Text("%s", T("Atmosphere", "大気"));
    constexpr int nAtm = sizeof(kAtmospheres)/sizeof(kAtmospheres[0]);
    const char* atmNames[nAtm];
    for (int i=0;i<nAtm;i++)
        atmNames[i] = (gLang == Lang::JA) ? kAtmNamesJA[i] : kAtmospheres[i].name;
    if (ImGui::Combo("##atm", &atmosphereIdx, atmNames, nAtm)) { /* drag recalculated each frame */ }
    ImGui::TextColored(ImVec4(0.58f,0.72f,0.88f,1.f),
        T("  rho=%.3f kg/m³", "  ρ=%.3f kg/m³"), kAtmospheres[atmosphereIdx].rho);

    } // CollapsingHeader Environment & Forces

    // ── Simulation controls ───────────────────────────────────────────────────
    ImGui::Spacing();
    {
        float bw = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
        constexpr float SBH = 32.f;
        if (gSimPaused) {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.22f,0.62f,0.38f,0.82f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f,0.70f,0.44f,0.92f));
            if (ImGui::Button(T("Play [P]", "再生[P]"), ImVec2(bw, SBH))) gSimPaused = false;
            ImGui::PopStyleColor(2);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.82f,0.52f,0.18f,0.82f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.92f,0.60f,0.24f,0.92f));
            if (ImGui::Button(T("Pause [P]", "停止[P]"), ImVec2(bw, SBH))) gSimPaused = true;
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine(0, 4);
        ImGui::BeginDisabled(!gSimPaused);
        if (ImGui::Button(T("Step [.]", "ステップ[.]"), ImVec2(bw, SBH))) gSimStep = true;
        ImGui::EndDisabled();
        ImGui::SameLine(0, 4);
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.76f,0.26f,0.30f,0.82f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f,0.34f,0.38f,0.94f));
        if (ImGui::Button(T("Reset", "リセット"), ImVec2(bw, SBH))) { resetBall(); gSimPaused = false; }
        ImGui::PopStyleColor(2);
        if (gSimPaused)
            ImGui::TextColored(ImVec4(0.95f,0.67f,0.24f,1.f), "%s",
                T("  PAUSED — press P or Step [.]", "  一時停止中 — P またはステップ [.]"));
    }
    ImGui::Spacing();

    // ── View Controls ────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader(T("View & Scene", "表示とシーン"), ImGuiTreeNodeFlags_DefaultOpen)) {
        float bwH = (ImGui::GetContentRegionAvail().x - 4) * 0.5f;
        char colorLabel[64];
        std::snprintf(colorLabel, sizeof(colorLabel),
            T("Grid: %s [C]", "色:%s[C]"), gridColors[currentColor].name);
        if (ImGui::Button(colorLabel, ImVec2(bwH, 0))) {
            currentColor = (currentColor + 1) % numColors;
            createGridMesh();
        }
        ImGui::SameLine(0, 4);
        if (ImGui::Button(T("Reset Camera [R]", "カメラ[R]"), ImVec2(bwH, 0)))
            yaw=35.f, pitch=25.f, cameraDistance=55.f;
        ImGui::Checkbox(T("Auto Rotate [Space]", "自動回転 [Space]"), &autoRotate);
        ImGui::TextDisabled(T("  Grid: %d   FPS: %.0f", "  グリッド: %d   FPS: %.0f"),
            gridSize, ImGui::GetIO().Framerate);

        // ── UI Theme selector ─────────────────────────────────────────────────
        ImGui::Spacing();
        ImGui::SeparatorText(T("UI Theme", "UIテーマ"));
        ImGui::Spacing();

        struct TI {
            const char* nameEN, *nameJA, *theoryEN, *theoryJA;
            ImVec4 dark, mid, bright; ImU32 accent;
        };
        static const TI kTI[5] = {
            {"Electric Blue", "エレクトリックブルー", "bold cyan gradient", "鮮やかなシアングラデーション",
             {.06f,.07f,.10f,1},{.20f,.40f,.60f,1},{.40f,.75f,1.00f,1}, IM_COL32(102,191,255,255)},
            {"Neon Purple",  "ネオンパープル", "vibrant violet", "鮮烈なバイオレット",
             {.07f,.06f,.10f,1},{.35f,.22f,.50f,1},{.75f,.45f,1.00f,1}, IM_COL32(191,115,255,255)},
            {"Lime Green",   "ライムグリーン", "bright emerald", "明るいエメラルド",
             {.06f,.09f,.07f,1},{.22f,.47f,.30f,1},{.45f,.95f,.60f,1}, IM_COL32(115,242,153,255)},
            {"Sunset Orange", "サンセットオレンジ", "warm gradient", "温かいグラデーション",
             {.10f,.07f,.06f,1},{.50f,.32f,.17f,1},{1.00f,.65f,.35f,1}, IM_COL32(255,165,89,255)},
            {"Hot Pink",     "ホットピンク", "bold magenta", "鮮やかなマゼンタ",
             {.10f,.06f,.09f,1},{.50f,.22f,.37f,1},{1.00f,.45f,.75f,1}, IM_COL32(255,115,191,255)},
        };

        float tCardW = ImGui::GetContentRegionAvail().x;
        ImDrawList* tDl = ImGui::GetWindowDrawList();
        for (int i = 0; i < 5; i++) {
            ImGui::PushID(100 + i);
            bool sel = (gTheme == i);
            constexpr float cardH = 56.f;
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##tc", ImVec2(tCardW, cardH))) gTheme = i;
            bool hov = ImGui::IsItemHovered();

            ImU32 bg  = sel ? IM_COL32(40,44,52,238)
                            : (hov ? IM_COL32(36,39,46,228) : IM_COL32(26,28,34,212));
            ImU32 brd = sel ? kTI[i].accent
                            : (hov ? IM_COL32(110,116,130,160) : IM_COL32(255,255,255,34));
            tDl->AddRectFilled(p0, {p0.x+tCardW, p0.y+cardH}, bg,  9.f);
            tDl->AddRect      (p0, {p0.x+tCardW, p0.y+cardH}, brd, 9.f, 0, sel ? 1.8f : 1.0f);

            float cy = p0.y+cardH*0.5f, r = 9.f, gap = r*2.4f, sx = p0.x+14.f+r;
            for (int s = 0; s < 3; s++) {
                ImVec4 cv = s==0 ? kTI[i].dark : s==1 ? kTI[i].mid : kTI[i].bright;
                tDl->AddCircleFilled({sx+s*gap, cy}, r, ImGui::ColorConvertFloat4ToU32(cv), 24);
                tDl->AddCircle      ({sx+s*gap, cy}, r, IM_COL32(255,255,255,30), 24);
            }
            float tx = sx + 3*gap + 10.f;
            tDl->AddText(nullptr, 0, {tx, p0.y+10.f},
                sel ? kTI[i].accent : IM_COL32(210,215,228,230),
                gLang==Lang::JA ? kTI[i].nameJA : kTI[i].nameEN);
            tDl->AddText(nullptr, 0, {tx, p0.y+28.f}, IM_COL32(138,148,168,175),
                gLang==Lang::JA ? kTI[i].theoryJA : kTI[i].theoryEN);
            if (sel) {
                float dx = p0.x+tCardW-16.f;
                tDl->AddCircleFilled({dx, cy}, 8.f, kTI[i].accent, 16);
                tDl->AddText(nullptr, 0, {dx-4.f, cy-6.f}, IM_COL32(0,0,0,220), "\xe2\x9c\x93");
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
    }

    ImGui::EndChild();
        ImGui::EndTabItem();
    } // end Environment Setting tab

    // ── Object Library tab ────────────────────────────────────────────────────
    if (ImGui::BeginTabItem(T("Library", "ライブラリ"))) {
    ImGui::BeginChild("##libscroll", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);

    // ── Category sidebar + card grid ──────────────────────────────────────────
    ImGui::BeginChild("##libcats", ImVec2(96,288), true);
    for (int ci=0; ci<kNumLibCats; ++ci) {
        bool sel = (gLibCatFilter==ci);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Header, alphaOf(themeAccent(), 0.32f));
        if (ImGui::Selectable((gLang==Lang::JA)?kLibCategoriesJA[ci]:kLibCategories[ci], sel, 0, ImVec2(0,22)))
            { gLibCatFilter=ci; gLibSelected=-1; gLibVariantSel=0; }
        if (sel) ImGui::PopStyleColor();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // card grid
    ImGui::BeginChild("##libgrid", ImVec2(0,288), true);
    ImDrawList* libDl = ImGui::GetWindowDrawList();
    constexpr float cardSz = 86.f;
    constexpr float cardPad = 8.f;
    int col=0; float startX = ImGui::GetCursorScreenPos().x;
    int availW = (int)ImGui::GetContentRegionAvail().x;
    int nCols = std::max(1, (int)((availW + cardPad) / (cardSz + cardPad)));

    for (int ei=0; ei<kNumLibEntries; ++ei) {
        const LibEntry& e = kLibEntries[ei];
        // category filter
        if (gLibCatFilter > 0 && std::strcmp(e.category, kLibCategories[gLibCatFilter])!=0)
            continue;

        if (col >= nCols) { ImGui::NewLine(); col=0; }
        ImVec2 cpos = ImGui::GetCursorScreenPos();
        bool selected = (gLibSelected == ei);
        ImVec4 accent = themeAccent();
        ImU32 bg   = selected ? toU32(alphaOf(accent, 0.24f)) : IM_COL32(34,37,44,220);
        ImU32 bord = selected ? toU32(alphaOf(accent, 0.88f)) : IM_COL32(255,255,255,36);
        libDl->AddRectFilled({cpos.x+1,cpos.y+2}, {cpos.x+cardSz+1, cpos.y+cardSz+2},
                             IM_COL32(0,0,0,35), 10.f);
        libDl->AddRectFilled(cpos, {cpos.x+cardSz, cpos.y+cardSz}, bg, 10.f);
        libDl->AddRect      (cpos, {cpos.x+cardSz, cpos.y+cardSz}, bord, 10.f, 0, selected ? 1.8f : 1.0f);

        // shape icon
        ImU32 iconFill = IM_COL32((int)(e.r*255),(int)(e.g*255),(int)(e.b*255),230);
        ImU32 iconEdge = IM_COL32(255,255,255,90);
        DrawLibIcon(libDl, cpos, cardSz, e.baseShape, iconFill, iconEdge);

        // name label (below icon)
        float textY = cpos.y + cardSz - 22.0f;
        libDl->PushClipRect({cpos.x+7, textY}, {cpos.x+cardSz-7, cpos.y+cardSz-5}, true);
        libDl->AddText({cpos.x+7, textY}, IM_COL32(232,235,240,245), (gLang==Lang::JA)?kLibEntryNamesJA[ei]:e.name);
        libDl->PopClipRect();

        // invisible button for click detection
        ImGui::SetCursorScreenPos(cpos);
        ImGui::PushID(ei);
        if (ImGui::InvisibleButton("##c", {cardSz,cardSz})) {
            gLibSelected = ei;
            gLibVariantSel = 0;
            const LibVariant& v = e.variants[0];
            gLibSx=v.sx; gLibSy=v.sy; gLibSz=v.sz;
            if (v.matHint>=0) gLibMat=v.matHint;
            gLibShape=e.baseShape;
        }
        if (ImGui::BeginDragDropSource()) {
            LibDragPayload dragData{ei, (gLibSelected == ei) ? gLibVariantSel : 0};
            ImGui::SetDragDropPayload("LIB_OBJECT", &dragData, sizeof(dragData));
            // Draw a floating icon + label at the cursor
            ImDrawList* fdl = ImGui::GetForegroundDrawList();
            ImVec2 mp = ImGui::GetMousePos();
            constexpr float prevSz = 54.f;
            ImU32 pfill = IM_COL32((int)(e.r*255),(int)(e.g*255),(int)(e.b*255),230);
            fdl->AddRectFilled({mp.x+12,mp.y+12},{mp.x+12+prevSz,mp.y+12+prevSz},
                               IM_COL32(34,37,44,230), 9.f);
            fdl->AddRect({mp.x+12,mp.y+12},{mp.x+12+prevSz,mp.y+12+prevSz},
                         toU32(alphaOf(themeAccent(), 0.90f)), 9.f, 0, 1.5f);
            DrawLibIcon(fdl, {mp.x+12,mp.y+12}, prevSz, e.baseShape,
                        pfill, IM_COL32(255,255,255,90));
            ImGui::Text("Drop to scene: %s", e.name);
            ImGui::EndDragDropSource();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s\n%s", e.name, e.desc);
            libDl->AddRect(cpos,{cpos.x+cardSz,cpos.y+cardSz},toU32(alphaOf(themeAccent(), 0.95f)),10.f,0,2.0f);
        }
        ImGui::PopID();

        if (col < nCols-1) { ImGui::SameLine(0,cardPad); }
        ++col;
    }
    ImGui::EndChild();

    // ── Variant picker ────────────────────────────────────────────────────────
    if (gLibSelected >= 0) {
        const LibEntry& e = kLibEntries[gLibSelected];
        ImGui::Separator();
        ImGui::TextColored(themeAccent(), "%s", (gLang==Lang::JA)?kLibEntryNamesJA[gLibSelected]:e.name);
        ImGui::SameLine(); ImGui::TextDisabled("  %s", e.desc);
        ImGui::Text("%s", T("Variants:", "バリアント:"));
        ImDrawList* vdl = ImGui::GetWindowDrawList();
        constexpr float vcSz=62.f;
        for (int vi=0; vi<e.nVariants; ++vi) {
            const LibVariant& v = e.variants[vi];
            bool vsel = (gLibVariantSel==vi);
            ImVec2 vp = ImGui::GetCursorScreenPos();
            ImU32 vbg = vsel ? toU32(alphaOf(themeAccent(), 0.24f)) : IM_COL32(34,37,44,215);
            ImU32 vbd = vsel ? toU32(alphaOf(themeAccent(), 0.88f)) : IM_COL32(255,255,255,38);
            vdl->AddRectFilled(vp,{vp.x+vcSz,vp.y+vcSz},vbg,8.f);
            vdl->AddRect(vp,{vp.x+vcSz,vp.y+vcSz},vbd,8.f,0,vsel ? 1.8f : 1.0f);
            // draw icon scaled by variant sx/sy/sz (squash/stretch hint)
            float maxS = std::max({v.sx,v.sy,v.sz});
            ImU32 vfill=IM_COL32((int)(e.r*255),(int)(e.g*255),(int)(e.b*255),220);
            DrawLibIcon(vdl, vp, vcSz, e.baseShape, vfill, IM_COL32(255,255,255,70));
            vdl->PushClipRect({vp.x+5, vp.y+vcSz-17}, {vp.x+vcSz-5, vp.y+vcSz-3}, true);
            vdl->AddText({vp.x+5, vp.y+vcSz-17}, IM_COL32(224,228,236,245), (gLang==Lang::JA)?kLibVariantNamesJA[gLibSelected][vi]:v.name);
            vdl->PopClipRect();
            ImGui::SetCursorScreenPos(vp);
            ImGui::PushID(100+vi);
            if (ImGui::InvisibleButton("##v",{vcSz,vcSz})) {
                gLibVariantSel=vi;
                gLibSx=v.sx; gLibSy=v.sy; gLibSz=v.sz;
                if (v.matHint>=0) gLibMat=v.matHint;
                gLibShape=e.baseShape;
            }
            if (ImGui::BeginDragDropSource()) {
                LibDragPayload dragData{gLibSelected, vi};
                ImGui::SetDragDropPayload("LIB_OBJECT", &dragData, sizeof(dragData));
                ImDrawList* fdl = ImGui::GetForegroundDrawList();
                ImVec2 mp = ImGui::GetMousePos();
                constexpr float prevSz = 54.f;
                ImU32 pfill = IM_COL32((int)(e.r*255),(int)(e.g*255),(int)(e.b*255),230);
                fdl->AddRectFilled({mp.x+12,mp.y+12},{mp.x+12+prevSz,mp.y+12+prevSz},
                                   IM_COL32(34,37,44,230), 9.f);
                fdl->AddRect({mp.x+12,mp.y+12},{mp.x+12+prevSz,mp.y+12+prevSz},
                             toU32(alphaOf(themeAccent(), 0.90f)), 9.f, 0, 1.5f);
                DrawLibIcon(fdl, {mp.x+12,mp.y+12}, prevSz, e.baseShape,
                            pfill, IM_COL32(255,255,255,90));
                ImGui::Text(T("Drop to scene: %s (%s)","シーンへドロップ: %s (%s)"),
                    (gLang==Lang::JA)?kLibEntryNamesJA[gLibSelected]:e.name,
                    (gLang==Lang::JA)?kLibVariantNamesJA[gLibSelected][vi]:v.name);
                ImGui::EndDragDropSource();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(T("%s — %s\nDrag to drop into scene","%s — %s\nシーンへドラッグ"),
                    (gLang==Lang::JA)?kLibEntryNamesJA[gLibSelected]:e.name,
                    (gLang==Lang::JA)?kLibVariantNamesJA[gLibSelected][vi]:v.name);
                vdl->AddRect(vp,{vp.x+vcSz,vp.y+vcSz},toU32(alphaOf(themeAccent(), 0.95f)),8.f,0,2.0f);
            }
            ImGui::PopID();
            if (vi < e.nVariants-1) ImGui::SameLine(0,4.f);
        }
        ImGui::Separator();

        // Scale fine-tune
        ImGui::TextDisabled("%s", T("SCALE OVERRIDE", "スケール調整"));
        ImGui::SetNextItemWidth(90); ImGui::SliderFloat("X##lsx",&gLibSx,0.1f,5.f,"%.2f"); ImGui::SameLine();
        ImGui::SetNextItemWidth(90); ImGui::SliderFloat("Y##lsy",&gLibSy,0.1f,5.f,"%.2f"); ImGui::SameLine();
        ImGui::SetNextItemWidth(90); ImGui::SliderFloat("Z##lsz",&gLibSz,0.1f,5.f,"%.2f");

        // Material + size
        ImGui::TextDisabled("%s", T("MATERIAL / SIZE", "材質 / サイズ"));
        ImGui::SetNextItemWidth(170);
        {constexpr int nM=sizeof(kObjectMaterials)/sizeof(kObjectMaterials[0]);
         const char* mn[nM]; for(int i=0;i<nM;i++) mn[i]=(gLang==Lang::JA)?kObjMatNamesJA[i]:kObjectMaterials[i].name;
         ImGui::Combo("##libmat2",&gLibMat,mn,nM);}
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##libsz2",&gLibSize,0.1f,3.0f,"r=%.2fm");

        // Effective size display
        const ObjectMaterial& lom = kObjectMaterials[gLibMat];
        ImGui::TextColored(ImVec4(0.7f,0.9f,0.7f,1.f),
            "  %.2f × %.2f × %.2f m   mass≈%.1f kg",
            2*gLibSize*gLibSx, 2*gLibSize*gLibSy, 2*gLibSize*gLibSz,
            lom.rho * shapeVolume(gLibShape, gLibSize, gLibSx, gLibSy, gLibSz));

        if (ImGui::Button(T("+ Add to Scene", "+ シーンに追加"), ImVec2(-1,0)))
            addSceneObject(gLibShape, gLibMat, gLibSize, gLibSx, gLibSy, gLibSz);
    } else {
        ImGui::TextDisabled("%s", T("  Click a card to select an object.",
                                    "  カードをクリックしてオブジェクトを選択。"));
    }

    ImGui::Separator();
    ImGui::TextDisabled(T("SCENE (%d objects, %d constraints)", "シーン (%d 個, %d 拘束)"),
                        (int)gSceneObjects.size(), (int)gSceneConstraints.size());
    for (int i=0;i<(int)gSceneObjects.size();++i) {
        bool sel=(i==gSelectedObjIdx);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Header, alphaOf(themeAccent(), 0.32f));
        if (ImGui::Selectable(gSceneObjects[i].label, sel)) gSelectedObjIdx=i;
        if (sel) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
            const SceneObject& so=gSceneObjects[i];
            const ObjectMaterial& som=kObjectMaterials[so.matIdx];
            ImGui::SetTooltip("%s  %s\nr=%.2f  scale %.2f×%.2f×%.2f\nv=%.2f m/s",
                kSceneShapeNames[so.shapeType], som.name,
                so.r, so.sx,so.sy,so.sz, std::sqrt(dot(so.vel,so.vel)));
        }
    }
    if (gSelectedObjIdx>=0 && gSelectedObjIdx<(int)gSceneObjects.size()) {
        if (ImGui::Button(T("Delete Selected", "選択を削除"), ImVec2(-1,0)))
            removeSceneObject(gSelectedObjIdx);
    }
    ImGui::EndChild();
        ImGui::EndTabItem();
    } // end Object Library tab

    // ── Object Workshop tab ───────────────────────────────────────────────────
    if (ImGui::BeginTabItem(T("Workshop", "ワークショップ"))) {

    if (ImGui::BeginTabBar("##wstabs")) {

        // ── CREATE tab ───────────────────────────────────────────────────────
        if (ImGui::BeginTabItem(T("Create", "作成"))) {
            ImGui::BeginChild("##crscroll", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);
            ImGui::TextDisabled("%s", T("BUILD FROM SCRATCH", "ゼロから構築"));
            ImGui::Text("%s", T("Base Shape", "基本形状"));
            ImGui::SetNextItemWidth(-1);
            ImGui::Combo("##wsshape",&gWsShape,getSceneShapeNamesL(),kNumSceneShapes);
            ImGui::Text("%s", T("Material", "材質"));
            ImGui::SetNextItemWidth(-1);
            {constexpr int nM=sizeof(kObjectMaterials)/sizeof(kObjectMaterials[0]);
             const char* mn[nM]; for(int i=0;i<nM;i++) mn[i]=(gLang==Lang::JA)?kObjMatNamesJA[i]:kObjectMaterials[i].name;
             ImGui::Combo("##wsmat",&gWsMat,mn,nM);}
            ImGui::Text("%s", T("Radius (base)", "半径（基本）"));
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##wsr",&gWsRadius,0.05f,5.0f,"%.3f m");
            ImGui::Separator();
            ImGui::TextDisabled("%s", T("NON-UNIFORM SCALE  (stretches & squashes)", "非等方スケール（伸縮）"));
            float colW = ImGui::GetContentRegionAvail().x / 3.0f - 4;
            ImGui::SetNextItemWidth(colW);
            ImGui::SliderFloat("X##wsx",&gWsSx,0.05f,8.f,"%.2f"); ImGui::SameLine(0,4);
            ImGui::SetNextItemWidth(colW);
            ImGui::SliderFloat("Y##wsy",&gWsSy,0.05f,8.f,"%.2f"); ImGui::SameLine(0,4);
            ImGui::SetNextItemWidth(colW);
            ImGui::SliderFloat("Z##wsz",&gWsSz,0.05f,8.f,"%.2f");
            if (ImGui::Button(T("Reset Scale##ws", "スケールリセット##ws"), ImVec2(0,0)))
                gWsSx=gWsSy=gWsSz=1.f;
            // Physics preview
            const ObjectMaterial& wom = kObjectMaterials[gWsMat];
            float wvol  = shapeVolume(gWsShape, gWsRadius, gWsSx, gWsSy, gWsSz);
            float wmass = wom.rho * wvol;
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f,0.9f,0.7f,1.f),
                T("Effective size: %.3f × %.3f × %.3f m", "実効サイズ: %.3f × %.3f × %.3f m"),
                2*gWsRadius*gWsSx, 2*gWsRadius*gWsSy, 2*gWsRadius*gWsSz);
            ImGui::TextColored(ImVec4(0.7f,0.9f,0.7f,1.f),
                T("Volume: %.4f m³   Mass: %.2f kg", "体積: %.4f m³   質量: %.2f kg"), wvol, wmass);
            ImGui::TextColored(ImVec4(1.0f,0.62f,0.05f,1.0f),
                T("rho=%.0f kg/m³   E=%.1f GPa   e=%.2f", "ρ=%.0f kg/m³   E=%.1f GPa   e=%.2f"),
                wom.rho, wom.E_GPa, wom.e_rest);
            float ra=gWsRadius*gWsSx, rb=gWsRadius*gWsSy, rc_=gWsRadius*gWsSz;
            ImGui::TextColored(ImVec4(0.68f,0.32f,0.87f,1.0f),
                "I_x=%.3f  I_y=%.3f  I_z=%.3f  kg·m²",
                wmass*(rb*rb+rc_*rc_)/5.f, wmass*(ra*ra+rc_*rc_)/5.f, wmass*(ra*ra+rb*rb)/5.f);
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.24f,0.68f,0.40f,0.86f));
            if (ImGui::Button(T("Spawn into Scene", "シーンに配置"), ImVec2(-1,0)))
                addSceneObject(gWsShape, gWsMat, gWsRadius, gWsSx, gWsSy, gWsSz);
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── CONNECT tab ──────────────────────────────────────────────────────
        if (ImGui::BeginTabItem(T("Connect", "接続"))) {
            ImGui::BeginChild("##conscroll", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);
            ImGui::TextDisabled("%s", T("CONSTRAIN TWO OBJECTS", "2物体の拘束"));
            int nObj = (int)gSceneObjects.size();
            ImGui::Text("%s", T("Object A", "物体 A"));
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##connA", gWsConnA>=0&&gWsConnA<nObj?gSceneObjects[gWsConnA].label:T("(none)","（なし）"))) {
                for (int i=0;i<nObj;++i) if (ImGui::Selectable(gSceneObjects[i].label, gWsConnA==i)) gWsConnA=i;
                ImGui::EndCombo();
            }
            ImGui::Text("%s", T("Object B  (none → world anchor)", "物体 B（なし → 世界固定点）"));
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##connB", gWsConnB>=0&&gWsConnB<nObj?gSceneObjects[gWsConnB].label:T("(world anchor)","（世界固定点）"))) {
                if (ImGui::Selectable(T("(world anchor)","（世界固定点）"), gWsConnB==-1)) gWsConnB=-1;
                for (int i=0;i<nObj;++i) if (ImGui::Selectable(gSceneObjects[i].label, gWsConnB==i)) gWsConnB=i;
                ImGui::EndCombo();
            }
            ImGui::Separator();
            ImGui::Text("%s", T("Constraint Type", "拘束種別"));
            ImGui::SetNextItemWidth(-1);
            ImGui::Combo("##contype",&gWsConType,getConstraintNamesL(),kNumConstraintTypes);

            ImGui::TextDisabled("%s", T("PIVOT A  (local offset from object A centre)",
                                        "ピボット A（物体 A の局所オフセット）"));
            float cw3 = ImGui::GetContentRegionAvail().x/3.f - 4;
            ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##pax",&gWsPivA.x,0,0,"%.2f"); ImGui::SameLine(0,4);
            ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##pay",&gWsPivA.y,0,0,"%.2f"); ImGui::SameLine(0,4);
            ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##paz",&gWsPivA.z,0,0,"%.2f");
            if (gWsConnB >= 0) {
                ImGui::TextDisabled("%s", T("PIVOT B  (local offset from object B centre)",
                                            "ピボット B（物体 B の局所オフセット）"));
                ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##pbx",&gWsPivB.x,0,0,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##pby",&gWsPivB.y,0,0,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw3); ImGui::InputFloat("##pbz",&gWsPivB.z,0,0,"%.2f");
            }
            if (gWsConType == 1) { // Hinge extras
                ImGui::TextDisabled("%s", T("HINGE AXIS & LIMITS", "ヒンジ軸と制限角"));
                ImGui::SetNextItemWidth(cw3); ImGui::SliderFloat("ax##ha",&gWsHingeAx.x,-1.f,1.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw3); ImGui::SliderFloat("ay##ha",&gWsHingeAx.y,-1.f,1.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw3); ImGui::SliderFloat("az##ha",&gWsHingeAx.z,-1.f,1.f,"%.2f");
                ImGui::SetNextItemWidth(130); ImGui::SliderFloat("Lo°##hl",&gWsLimitLow,-180.f,0.f,"%.0f"); ImGui::SameLine();
                ImGui::SetNextItemWidth(130); ImGui::SliderFloat("Hi°##hh",&gWsLimitHigh,0.f,180.f,"%.0f");
            }
            if (gWsConType == 2) { // Spring extras
                ImGui::TextDisabled("%s", T("SPRING PARAMETERS", "バネパラメータ"));
                ImGui::SetNextItemWidth(-1); ImGui::SliderFloat(T("Stiffness k##sk","剛性 k##sk"),&gWsSpringK,1.f,50000.f,"%.0f N/m",ImGuiSliderFlags_Logarithmic);
                ImGui::SetNextItemWidth(-1); ImGui::SliderFloat(T("Damping d##sd","減衰 d##sd"),&gWsSpringD,0.f,2000.f,"%.0f N·s/m");
                float wn = (gWsConnA>=0&&gWsConnA<nObj) ? std::sqrt(gWsSpringK / std::max(0.001f,
                    kObjectMaterials[gSceneObjects[gWsConnA].matIdx].rho *
                    4.f/3.f*pi*gSceneObjects[gWsConnA].r*gSceneObjects[gWsConnA].r*gSceneObjects[gWsConnA].r)) : 0.f;
                if (wn>0) ImGui::TextColored(ImVec4(1.0f,0.80f,0.0f,1.0f),"  ω_n ≈ %.2f rad/s   f_n ≈ %.2f Hz",wn,wn/(2.f*pi));
            }
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Button, alphaOf(themeAccent(), 0.88f));
            if (ImGui::Button(T("Connect##do", "接続##do"), ImVec2(-1,0)) && gWsConnA>=0 && gBtWorld) {
                SceneConstraint sc;
                sc.typeIdx=gWsConType; sc.objA=gWsConnA; sc.objB=gWsConnB;
                sc.pivotA=gWsPivA; sc.pivotB=gWsPivB;
                sc.hingeAxis=gWsHingeAx;
                sc.limitLow=gWsLimitLow; sc.limitHigh=gWsLimitHigh;
                sc.springK=gWsSpringK; sc.springD=gWsSpringD;
                addConstraint(sc);
            }
            ImGui::PopStyleColor();

            if (!gSceneConstraints.empty()) {
                ImGui::Separator();
                ImGui::TextDisabled("%s", T("ACTIVE CONSTRAINTS", "有効な拘束"));
                for (int ci=0;ci<(int)gSceneConstraints.size();++ci) {
                    auto& sc=gSceneConstraints[ci];
                    char lbl[80];
                    std::snprintf(lbl,sizeof(lbl),"[%d] %s",ci,sc.label);
                    ImGui::Text("%s", lbl);
                    ImGui::SameLine();
                    ImGui::PushID(ci+2000);
                    if (ImGui::SmallButton(T("Remove", "削除"))) removeConstraint(ci);
                    ImGui::PopID();
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        // ── REFINE tab ───────────────────────────────────────────────────────
        if (ImGui::BeginTabItem(T("Refine", "詳細設定"))) {
            ImGui::BeginChild("##refscroll", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);
            if (gSelectedObjIdx>=0 && gSelectedObjIdx<(int)gSceneObjects.size()) {
                SceneObject& obj = gSceneObjects[gSelectedObjIdx];
                ImGui::TextColored(themeAccent(),
                    T("Editing: %s", "編集中: %s"), obj.label);
                ImGui::InputText(T("Name##rn", "名称##rn"), obj.label, sizeof(obj.label));

                ImGui::Separator();
                ImGui::TextDisabled("%s", T("POSITION", "位置"));
                bool pc=false;
                ImGui::SetNextItemWidth(-1); pc|=ImGui::SliderFloat("##rpx",&obj.pos.x,-500.f,500.f,"X: %.2f m");
                ImGui::SetNextItemWidth(-1); pc|=ImGui::SliderFloat("##rpy",&obj.pos.y,  0.f, 60.f,"Y: %.2f m");
                ImGui::SetNextItemWidth(-1); pc|=ImGui::SliderFloat("##rpz",&obj.pos.z,-500.f,500.f,"Z: %.2f m");
                if (pc) pushSceneObjectToBullet(obj);

                ImGui::Separator();
                ImGui::TextDisabled("%s", T("ROTATION (degrees)", "回転（度）"));
                bool rc2=false;
                ImGui::SetNextItemWidth(-1); rc2|=ImGui::SliderFloat("##rep",&obj.euler.x,-180.f,180.f,T("Pitch: %.1f","ピッチ: %.1f"));
                ImGui::SetNextItemWidth(-1); rc2|=ImGui::SliderFloat("##rey",&obj.euler.y,-180.f,180.f,T("Yaw:   %.1f","ヨー: %.1f"));
                ImGui::SetNextItemWidth(-1); rc2|=ImGui::SliderFloat("##rer",&obj.euler.z,-180.f,180.f,T("Roll:  %.1f","ロール: %.1f"));
                if (rc2) { obj.orient=eulerToQuat(obj.euler.x,obj.euler.y,obj.euler.z); pushSceneObjectToBullet(obj); }

                ImGui::Separator();
                ImGui::TextDisabled("%s", T("SCALE (stretches physics shape)", "スケール（物理形状を変形）"));
                float cw = ImGui::GetContentRegionAvail().x/3.f - 4;
                float nsx=obj.sx, nsy=obj.sy, nsz=obj.sz;
                bool scChanged=false;
                ImGui::SetNextItemWidth(cw); scChanged|=ImGui::SliderFloat("X##rsx",&nsx,0.05f,8.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw); scChanged|=ImGui::SliderFloat("Y##rsy",&nsy,0.05f,8.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw); scChanged|=ImGui::SliderFloat("Z##rsz",&nsz,0.05f,8.f,"%.2f");
                if (scChanged) {
                    obj.sx=nsx; obj.sy=nsy; obj.sz=nsz;
                    if (obj.bshape) obj.bshape->setLocalScaling(btVector3(nsx,nsy,nsz));
                }
                float newR=obj.r;
                ImGui::SetNextItemWidth(-1);
                if (ImGui::SliderFloat("##rsr",&newR,0.1f,5.f,T("Radius: %.3f m","半径: %.3f m")) && newR!=obj.r) {
                    obj.r=newR; rebuildSceneObjectShape(gSelectedObjIdx);
                }

                ImGui::Separator();
                ImGui::TextDisabled("%s", T("MATERIAL", "材質"));
                ImGui::SetNextItemWidth(-1);
                {constexpr int nM=sizeof(kObjectMaterials)/sizeof(kObjectMaterials[0]);
                 const char* mn[nM]; for(int i=0;i<nM;i++) mn[i]=(gLang==Lang::JA)?kObjMatNamesJA[i]:kObjectMaterials[i].name;
                 if (ImGui::Combo("##rmat",&obj.matIdx,mn,nM)) rebuildSceneObjectShape(gSelectedObjIdx);}

                ImGui::Separator();
                ImGui::TextDisabled("%s", T("FORCE CONTROL", "力の制御"));
                ImGui::Checkbox(T("Continuous force##rf", "持続力##rf"), &obj.forceOn);
                float cw2=ImGui::GetContentRegionAvail().x/3.f-4;
                ImGui::SetNextItemWidth(cw2); ImGui::SliderFloat("dX##fdx",&obj.forceDir.x,-1.f,1.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw2); ImGui::SliderFloat("dY##fdy",&obj.forceDir.y,-1.f,1.f,"%.2f"); ImGui::SameLine(0,4);
                ImGui::SetNextItemWidth(cw2); ImGui::SliderFloat("dZ##fdz",&obj.forceDir.z,-1.f,1.f,"%.2f");
                ImGui::SetNextItemWidth(-1); ImGui::SliderFloat(T("Magnitude (N)##fmag","大きさ (N)##fmag"),&obj.forceMag,0.f,100000.f,"%.0f N",ImGuiSliderFlags_Logarithmic);
                ImGui::SetNextItemWidth(-1);
                ImGui::SliderFloat(T("Impulse##impmag","力積##impmag"),&obj.impulseStrength,1.f,500000.f,"%.0f N·s",ImGuiSliderFlags_Logarithmic);
                char fireL[64];
                std::snprintf(fireL,sizeof(fireL),T("Fire Impulse (%.0f N·s)","力積発射 (%.0f N·s)"),obj.impulseStrength);
                if (ImGui::Button(fireL, ImVec2(-1,0)) && obj.body) {
                    Vec3 fd=normalize(obj.forceDir);
                    obj.body->applyCentralImpulse(btVector3(fd.x*obj.impulseStrength,fd.y*obj.impulseStrength,fd.z*obj.impulseStrength));
                    obj.body->activate();
                }
                ImGui::Separator();
                if (ImGui::Button(T("Reset Velocity##rv", "速度リセット##rv"), ImVec2(0,0)) && obj.body) {
                    obj.body->setLinearVelocity(btVector3(0,0,0));
                    obj.body->setAngularVelocity(btVector3(0,0,0));
                    obj.body->clearForces(); obj.body->activate();
                }
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.7f,0.9f,0.7f,1.f),"v=%.2f m/s", std::sqrt(dot(obj.vel,obj.vel)));

                ImGui::Separator();
                if (ImGui::Button(T("Clone Object##clone", "複製##clone"), ImVec2(-1,0)))
                    addSceneObject(obj.shapeType, obj.matIdx, obj.r, obj.sx, obj.sy, obj.sz);
                if (ImGui::Button(T("Delete Object##del", "削除##del"), ImVec2(-1,0)))
                    removeSceneObject(gSelectedObjIdx);
            } else {
                ImGui::TextDisabled("%s", T("  Select an object in the scene list to refine.",
                                            "  シーンリストから物体を選択して詳細設定。"));
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
        ImGui::EndTabItem();
    } // end Object Workshop tab

    ImGui::EndTabBar();
} // end main tab bar
    ImGui::End(); // end ##mainpanel
} // end if (gPanelOpen)

    // Pop global style
    ImGui::PopStyleColor(15);
    ImGui::PopStyleVar(4);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void mainLoopIteration() {
#if defined(__EMSCRIPTEN__)
    ResizeCanvasForDPI();
#endif
    double currentTime = glfwGetTime();
    float dt = static_cast<float>(currentTime - previousTime);
    previousTime = currentTime;
    if (dt > 0.1f) dt = 0.1f;   // safety clamp

    bool doPhysics = !gSimPaused || gSimStep;
    gSimStep = false;   // consume the step token regardless

    if (doPhysics) {
        updatePhysics(dt);
    }

    if (autoRotate) yaw = std::fmod(yaw + dt * 25.0f, 360.0f);
    render();
    renderHUD();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
}

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "Could not initialize GLFW.\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);  // full Retina pixel density

    GLFWwindow* window = glfwCreateWindow(900, 700, "Physics Experiment", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Could not create an OpenGL context window.\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    gWindow = window;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
#if defined(__EMSCRIPTEN__)
    ResizeCanvasForDPI();
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenUiEvent*, void*) -> EM_BOOL {
            ResizeCanvasForDPI();
            return EM_TRUE;
        });
#endif
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
#if !defined(__EMSCRIPTEN__)
    glfwSetCursorPosCallback(window, cursorCallback);
#endif
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glEnable(GL_DEPTH_TEST);
#if !defined(__EMSCRIPTEN__)
    // WebGL2 has no runtime MSAA toggle — antialiasing is requested at
    // context creation instead (via the GLFW_SAMPLES hint above, which
    // GLFW's Emscripten port maps onto the canvas context's antialias flag).
    glEnable(GL_MULTISAMPLE);
#endif
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ImGui — Retina-aware + 3D style
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
#if defined(__EMSCRIPTEN__)
    // Rule out stale saved window state as a factor while diagnosing the
    // panel-position bug — every window here already passes
    // ImGuiWindowFlags_NoSavedSettings, but this removes any doubt.
    ImGui::GetIO().IniFilename = nullptr;
#endif

    // Query display content scale (2.0 on Retina, 1.0 on standard).
#if defined(__EMSCRIPTEN__)
    // glfwGetWindowContentScale would report framebuffer-size ÷ GLFW's
    // internal "window size" (still the literal 900x700 from
    // glfwCreateWindow, since ResizeCanvasForDPI resizes the canvas
    // directly rather than going through glfwSetWindowSize) — that ratio
    // no longer means anything once the canvas has been resized out from
    // under it, so read the real device pixel ratio directly instead.
    float dpiScale = (float)emscripten_get_device_pixel_ratio();
#else
    float dpiX = 1.0f, dpiY = 1.0f;
    glfwGetWindowContentScale(window, &dpiX, &dpiY);
    float dpiScale = dpiX;
#endif

    ImGuiIO& io = ImGui::GetIO();

    const float baseSize = 15.0f;
    const float pixelSize = baseSize * dpiScale;
    ImFontConfig cfg;
    cfg.OversampleH = 3;
    cfg.OversampleV = 3;
    cfg.PixelSnapH  = false;

#if defined(__EMSCRIPTEN__)
    // Apple's system fonts can't legally be bundled into a public web build,
    // and don't exist in Emscripten's virtual filesystem anyway. DejaVu Sans
    // (Bitstream Vera license, redistribution explicitly permitted) is
    // preloaded instead — a real vector TTF renders crisply at any size,
    // unlike ImGui's tiny built-in bitmap font stretched up, which is both
    // blurry and has different glyph widths than the UI was laid out for.
    ImFont* fontMain = io.Fonts->AddFontFromFileTTF("/fonts/DejaVuSans.ttf", pixelSize, &cfg);
    if (!fontMain) {
        ImFontConfig fb; fb.SizePixels = pixelSize;
        io.Fonts->AddFontDefault(&fb);
    }

    // Merge Japanese glyphs from Noto Sans JP (SIL OFL — Google's own font,
    // explicitly released for exactly this kind of embedding). DejaVu Sans
    // has no CJK coverage on its own.
    {
        ImFontConfig cfgJP;
        cfgJP.MergeMode   = true;
        cfgJP.OversampleH = 2;
        cfgJP.OversampleV = 2;
        const ImWchar* jpRanges = io.Fonts->GetGlyphRangesJapanese();
        io.Fonts->AddFontFromFileTTF("/fonts/NotoSansJP-Regular.otf", pixelSize, &cfgJP, jpRanges);
    }
#else
    // Load San Francisco (macOS native UI font) — crisp at any size
    ImFont* fontSF = io.Fonts->AddFontFromFileTTF(
        "/System/Library/Fonts/SFNS.ttf", pixelSize, &cfg);
    if (!fontSF) {
        // Fallback: built-in bitmap font scaled up
        ImFontConfig fb; fb.SizePixels = pixelSize;
        io.Fonts->AddFontDefault(&fb);
    }

    // Merge Japanese glyphs so kanji/kana render correctly
    {
        ImFontConfig cfgJP;
        cfgJP.MergeMode    = true;
        cfgJP.OversampleH  = 2;
        cfgJP.OversampleV  = 2;
        const ImWchar* jpRanges = io.Fonts->GetGlyphRangesJapanese();
        // Try known macOS CJK font paths in order
        const char* jpPaths[] = {
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/Library/Fonts/Arial Unicode MS.ttf",
            nullptr
        };
        for (int pi = 0; jpPaths[pi]; ++pi) {
            if (io.Fonts->AddFontFromFileTTF(jpPaths[pi], pixelSize, &cfgJP, jpRanges))
                break;
        }
    }
#endif

    io.FontGlobalScale = 1.0f / dpiScale;

    ImGui::StyleColorsDark();
    setup3DStyle(dpiScale);

    ImGui_ImplGlfw_InitForOpenGL(window, false);
#if defined(__EMSCRIPTEN__)
    // Without a registered cursor-enter callback, ImGui_ImplGlfw_UpdateMouseData()
    // treats bd->MouseWindow as never set and, every single frame, overwrites
    // whatever position our own event-driven callbacks report with a fresh
    // glfwGetCursorPos() read — which is in GLFW's stale internal window-size
    // space (see ApplyWebDisplayMetrics), not the live CSS space everything
    // else here uses. Marking the mouse as having "entered" once sets
    // bd->MouseWindow and permanently disables that per-frame poll, leaving
    // our raw mousemove/touch callbacks as the sole source of io.MousePos.
    ImGui_ImplGlfw_CursorEnterCallback(window, 1);
    ImGui_ImplOpenGL3_Init("#version 300 es");
    // GLFW's Emscripten port doesn't reliably normalize trackpad "smooth
    // scroll" wheel events (DOM_DELTA_PIXEL), so two-finger scrolling either
    // did nothing or barely moved anything through ImGui_ImplGlfw_ScrollCallback.
    // Reading the browser wheel event directly and normalizing by its own
    // deltaMode sidesteps that entirely.
    emscripten_set_wheel_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenWheelEvent* e, void*) -> EM_BOOL {
            float scale = e->deltaMode == DOM_DELTA_PIXEL ? 0.01f
                        : e->deltaMode == DOM_DELTA_LINE  ? 1.0f
                                                           : 20.0f; // DOM_DELTA_PAGE
            ImGui::GetIO().AddMouseWheelEvent(
                (float)-e->deltaX * scale, (float)-e->deltaY * scale);
            return EM_TRUE;
        });

    // GLFW's Emscripten port has a "CSS scaling" feature that rescales raw
    // mouse coordinates by (GLFW's internal window size / the canvas's CSS
    // rect size) before handing them to glfwSetCursorPosCallback. That
    // internal window size is fixed at whatever glfwCreateWindow was called
    // with and never updated by ResizeCanvasForDPI's manual canvas resizing,
    // so on any display where the CSS size differs from that, GLFW reports
    // mouse positions in the wrong coordinate space — offset from where
    // ImGui (using the live CSS-based io.DisplaySize) actually draws things.
    // Reading the raw target-relative coordinates directly, the same way
    // touch input already does below, sidesteps that scaling entirely.
    emscripten_set_mousemove_callback("#canvas", nullptr, true,
        [](int, const EmscriptenMouseEvent* e, void*) -> EM_BOOL {
            cursorCallback(gWindow, e->targetX, e->targetY);
            return EM_TRUE;
        });

    // Touch support — there's no mouse/trackpad on a phone, so without this
    // the app loads but is completely uninteractive there. One finger reuses
    // the existing mouse callbacks directly (so gizmo drag / object pick /
    // camera orbit / ImGui widgets all keep working exactly as they already
    // do for a mouse); two fingers pinch-zoom, which has no mouse equivalent
    // to reuse and needs its own tracking.
    static double sPinchStartDist = 0.0;
    static float  sPinchStartCameraDistance = 0.0;
    static bool   sTouchDragging = false;

    emscripten_set_touchstart_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenTouchEvent* e, void*) -> EM_BOOL {
            if (e->numTouches == 1) {
                cursorCallback(gWindow, e->touches[0].targetX, e->touches[0].targetY);
                mouseButtonCallback(gWindow, GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS, 0);
                sTouchDragging = true;
            } else if (e->numTouches == 2) {
                if (sTouchDragging) {
                    mouseButtonCallback(gWindow, GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, 0);
                    sTouchDragging = false;
                }
                double dx = e->touches[0].targetX - e->touches[1].targetX;
                double dy = e->touches[0].targetY - e->touches[1].targetY;
                sPinchStartDist = std::sqrt(dx * dx + dy * dy);
                sPinchStartCameraDistance = cameraDistance;
            }
            return EM_TRUE;
        });

    emscripten_set_touchmove_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenTouchEvent* e, void*) -> EM_BOOL {
            if (e->numTouches == 1 && sTouchDragging) {
                cursorCallback(gWindow, e->touches[0].targetX, e->touches[0].targetY);
            } else if (e->numTouches == 2 && sPinchStartDist > 1.0) {
                double dx = e->touches[0].targetX - e->touches[1].targetX;
                double dy = e->touches[0].targetY - e->touches[1].targetY;
                double dist = std::sqrt(dx * dx + dy * dy);
                cameraDistance = std::clamp(
                    sPinchStartCameraDistance * (float)(sPinchStartDist / dist), 3.0f, 2000.0f);
            }
            return EM_TRUE;
        });

    emscripten_set_touchend_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenTouchEvent* e, void*) -> EM_BOOL {
            if (e->numTouches == 0) {
                if (sTouchDragging) {
                    mouseButtonCallback(gWindow, GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, 0);
                    sTouchDragging = false;
                }
                sPinchStartDist = 0.0;
            }
            return EM_TRUE;
        });
    emscripten_set_touchcancel_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true,
        [](int, const EmscriptenTouchEvent*, void*) -> EM_BOOL {
            if (sTouchDragging) {
                mouseButtonCallback(gWindow, GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, 0);
                sTouchDragging = false;
            }
            sPinchStartDist = 0.0;
            return EM_TRUE;
        });
#else
    ImGui_ImplOpenGL3_Init("#version 410");
#endif

    createShaderProgram();
    createTerrainMesh();
    createGridMesh();
    createAxesMesh();
    createUnitSphereMesh();
    createUnitBoxMesh();
    createCylinderMesh();
    createConeMesh();
    createCapsuleMesh();
    createCarMesh();
    buildBulletWorld();
    printHelp();

    previousTime = glfwGetTime();

#if defined(__EMSCRIPTEN__)
    // A blocking while-loop never returns control to the browser's own
    // event loop, which freezes the tab ("Page Unresponsive") instead of
    // rendering anything. emscripten_set_main_loop hands frame-by-frame
    // control back via requestAnimationFrame instead — the standard way
    // to port a native run-loop to the web. With simulate_infinite_loop=1
    // this never returns, so the shutdown/cleanup below only runs in the
    // native build (fine — the page just keeps looping until closed).
    emscripten_set_main_loop(mainLoopIteration, 0, 1);
#else
    while (!glfwWindowShouldClose(window)) {
        mainLoopIteration();
    }
    destroyBulletWorld();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
#endif
    return EXIT_SUCCESS;
}
