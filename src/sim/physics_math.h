#pragma once
// sim/physics_math.h — Pure physics formulas.
// No Bullet, OpenGL, or ImGui dependencies. Include anywhere, unit-testable.
//
// Each formula cites the physical model it implements and any calibration
// constants, so results can be verified against published references.

#include "materials.h"
#include <cmath>
#include <algorithm>

// Reduced (combined) Young's modulus for Hertz contact between two elastic bodies.
// E* = 1 / [ (1-ν₁²)/E₁ + (1-ν₂²)/E₂ ]    [Pa]
inline float computeReducedModulus(int omIdx, int gmIdx) {
    const ObjectMaterial& om = kObjectMaterials[omIdx];
    const GroundMaterial&  gm = kGroundMaterials[gmIdx];
    float E1 = om.E_GPa * 1e9f, nu1 = om.nu;
    float E2 = gm.E_GPa * 1e9f, nu2 = gm.nu;
    float inv = (1.0f - nu1*nu1) / E1 + (1.0f - nu2*nu2) / E2;
    return 1.0f / std::max(inv, 1.0e-20f);
}

// Estimated critical normal impact velocity for brittle fracture (Hertz + Griffith).
// v_c = C · √(K_Ic⁴ / (ρ·E*·σ_t²·R))    [m/s]
// C = 22.5 (calibrated constant; document if used for research).
// Returns 0 for ductile materials (σ_y > 0) — brittle model not applicable.
inline float computeCriticalFractureVelocity(int omIdx, int gmIdx, float R) {
    const ObjectMaterial& om = kObjectMaterials[omIdx];
    if (om.sigma_y > 0.0f) return 0.0f;
    float sigma_t = om.sigma_t * 1e6f;
    float K_Ic    = om.K_Ic   * 1e6f;
    float rho     = om.rho;
    float Estar   = computeReducedModulus(omIdx, gmIdx);
    float denom   = rho * Estar * sigma_t * sigma_t * R;
    if (denom < 1.0e-30f) return 1.0e9f;
    float K4 = K_Ic * K_Ic * K_Ic * K_Ic;
    return 22.5f * std::sqrt(K4 / denom);
}

// Mean fragment count (Grady–Kipp energy-balance model).
// n ≈ ∛(E_scatter / (G_c · A))    with G_c = K_Ic²/E₁
inline int computeFragmentCount(int omIdx, float R, float E_scatter) {
    const ObjectMaterial& om = kObjectMaterials[omIdx];
    float K_Ic = om.K_Ic * 1.0e6f;
    float E1   = om.E_GPa * 1.0e9f;
    float G_c  = (K_Ic * K_Ic) / E1;
    constexpr float pi_val = 3.14159265358979323846f;
    float A    = 4.0f * pi_val * R * R;
    float ratio = E_scatter / (G_c * A);
    if (ratio < 0.01f) return 5;
    float n = std::cbrt(ratio) * 2.5f;
    return std::clamp((int)n, 5, 24);
}

// Geometrically correct volume for each collision shape with non-uniform scale.
// r is the base radius; sx/sy/sz are dimensionless scale factors so the
// actual half-extents are (r·sx, r·sy, r·sz).
inline float shapeVolume(int shapeType, float r, float sx, float sy, float sz) {
    constexpr float pi_val = 3.14159265358979323846f;
    float rx = r*sx, ry = r*sy, rz = r*sz;
    switch (shapeType) {
        case 0: return (4.0f/3.0f) * pi_val * rx * ry * rz;      // ellipsoid
        case 1: return 8.0f * rx * ry * rz;                       // box  2rx × 2ry × 2rz
        case 2: return pi_val * rx * rz * (2.0f * ry);            // cylinder  πR²h
        case 3: { float rXZ=(rx+rz)*0.5f;
                  return (pi_val/3.0f) * rXZ*rXZ * (2.0f*ry); }  // cone  πR²h/3
        case 4: { float rXZ=(rx+rz)*0.5f;
                  return pi_val * rXZ*rXZ * ry                    // capsule: cylinder
                       + (4.0f/3.0f) * pi_val * rXZ*rXZ*rXZ; }   //        + sphere caps
        case 5: return 8.0f * rx * ry * rz;                       // car — box volume same as case 1
        default: return (4.0f/3.0f) * pi_val * rx * ry * rz;
    }
}
