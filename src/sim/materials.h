#pragma once
// sim/materials.h — Physics material database.
// Pure constexpr data: no Bullet, no OpenGL, no ImGui dependencies.
// Testable independently with any C++17 compiler.

struct ObjectMaterial {
    const char* name;
    float rho;        // density           kg/m³
    float E_GPa;      // Young's modulus    GPa
    float nu;         // Poisson's ratio
    float sigma_t;    // tensile strength   MPa
    float sigma_y;    // yield strength     MPa  (0 = fully brittle)
    float K_Ic;       // fracture toughness MPa·m^0.5
    float c_L;        // longitudinal sound speed  m/s
    float mu_fric;    // surface friction   (Bullet)
    float e_rest;     // restitution        (Bullet, from COR tables)
    float mu_roll;    // rolling friction
    float r, g, b;    // display colour
};

struct GroundMaterial {
    const char* name;
    float E_GPa;      // Young's modulus GPa
    float nu;         // Poisson's ratio
    float mu_fric;
    float e_rest;
    float mu_roll;
};

struct GravityEnv  { const char* name; float g; };
struct AtmosphereType { const char* name; float rho; float mu; };

// ─────────────────────────────────────────────────────────────────────────────
// Object materials — representative handbook-style values.
// Not exact for every alloy/sample; treat as calibrated defaults.
//  name            ρ       E      ν     σ_t   σ_y   K_Ic  c_L    μ_f   e     μ_r    R     G     B
constexpr ObjectMaterial kObjectMaterials[] = {
{"Iron   (Fe-56)", 7874, 211.0f,0.26f, 200.f, 210.f,50.0f,5120, 0.65f,0.68f,0.002f,0.60f,0.61f,0.65f},
{"Copper (Cu-63)", 8960, 128.0f,0.34f, 220.f,  70.f,100.f,3810, 0.55f,0.68f,0.002f,0.72f,0.45f,0.20f},
{"Alum.  (Al-27)", 2700,  70.0f,0.33f,  90.f, 270.f, 35.f,5100, 0.62f,0.80f,0.002f,0.85f,0.85f,0.90f},
{"Lead   (Pb-208)",11340, 16.0f,0.44f,  17.f,  11.f,  2.f,1190, 0.62f,0.05f,0.004f,0.35f,0.35f,0.40f},
{"Titanium(Ti-48)",4510, 116.0f,0.32f, 240.f, 880.f, 70.f,5080, 0.55f,0.74f,0.002f,0.52f,0.54f,0.58f},
{"Wood   (Oak)",    700,  12.0f,0.37f,  80.f,   0.f,  8.0f,4150, 0.62f,0.66f,0.005f,0.55f,0.37f,0.18f},
{"Rubber",         1500,   0.05f,0.50f, 30.f,   0.f,  0.1f, 180, 1.00f,0.95f,0.020f,0.12f,0.12f,0.12f},
{"Ice    (H2O)",    917,   9.8f,0.33f,   2.f,   0.f,  0.2f,3900, 0.06f,0.37f,0.001f,0.75f,0.90f,0.98f},
};
constexpr int kNumObjectMaterials = (int)(sizeof(kObjectMaterials)/sizeof(kObjectMaterials[0]));

// Ground materials — name, E_GPa, ν, μ_f, e_rest, μ_roll
// ─ E, ν  : used in Hertz E* (computeReducedModulus).
// ─ μ_f   : Bullet multiplies object×ground friction; values chosen so the
//            pair product matches published sliding-friction tables.
// ─ e_rest: Bullet multiplies object×ground; values give correct pair COR.
// ─ μ_roll: rolling friction coefficient.
// Sources: Shigley's MED 10e, Roark's 8e, ASTM handbooks, Bowles Soil Mech 5e.
constexpr GroundMaterial kGroundMaterials[] = {
// name           E_GPa   ν      μ_f    e_rest  μ_roll
{"Steel plate",  210.0f, 0.29f, 0.65f, 0.95f, 0.002f},
// EN 10025 structural steel: E=210 GPa, ν=0.29 (Shigley's 10e §2).
// e=0.95 → iron×steel pair = 0.68×0.95 = 0.65 (matches Goldsmith 1960 impact tables).

{"Concrete",      32.0f, 0.20f, 0.70f, 0.55f, 0.010f},
// Normal-strength concrete C25/C30: E=29–33 GPa (EN 1992); ν=0.20 standard.
// μ_f=0.70 → iron×concrete = 0.46; matches ACI 318 steel-on-concrete (0.4–0.6).
// e=0.55 → iron×concrete pair = 0.374 (Stronge 2000: 0.3–0.5 for metal on concrete).

{"Soil (dry)",    0.05f, 0.35f, 0.65f, 0.30f, 0.060f},
// Deformation modulus 50 MPa: medium-dense dry cohesionless soil (Bowles 5e §5).
// ν=0.35 typical for dry silt/sand (Das Principles of Geotechnical Eng. 9e §15).
// μ_f=0.65 → iron×soil pair = 0.42 (soil-metal sliding: 0.35–0.55, Bowles).
// μ_roll=0.060: ball rolling on compacted dry soil (Bekker 1969 terramechanics).

{"Wood floor",   12.0f, 0.35f, 0.70f, 0.65f, 0.008f},
// Oak hardwood along grain: E≈11–13 GPa, ν≈0.35 (Wood Handbook USDA 2021 ch.4).
// e=0.65 → iron×wood = 0.44 (consistent with Sondergaard et al. 1990 measurements).

{"Ice sheet",     9.8f, 0.33f, 0.10f, 0.55f, 0.002f},
// Polycrystalline ice at 0 °C: E=9–10 GPa, ν=0.33 (Hobbs 1974; Petrenko & Whitworth).
// μ_f=0.10: ice kinetic friction (Bowden & Tabor 1950; varies 0.03–0.15 with speed).
// e=0.55 → iron×ice pair = 0.374 (Gugan 2000: metal sphere on ice 0.3–0.5).
// μ_roll=0.002: WCF curling ice standard; 0.001 is too ideal.

{"Sand",          0.03f, 0.33f, 0.65f, 0.18f, 0.100f},
// Loose dry sand: E≈15–50 MPa (Bowles §5, CPT correlations), ν=0.30–0.35.
// μ_f=0.65 → iron×sand pair = 0.42 (friction angle φ≈30° → tan φ≈0.58, Bowles).
// e=0.18 → iron×sand pair = 0.12 (Mao et al. 2004: ball impact on granular bed).
// μ_roll=0.100: Bekker 1969 loose-sand rolling resistance (0.06–0.20 range).

{"Rubber mat",    0.005f,0.49f, 1.00f, 0.80f, 0.030f},
// Vulcanised natural rubber mat: E≈1–10 MPa (Mark Polymer Data Handbook §2).
// 0.10 GPa (100 MPa) is hard ebonite — not a mat.
// ν=0.49 (rubber is nearly incompressible; ν=0.50 exact → (1−ν²)/E → singularity in E*).
// e=0.80 → rubber×mat = 0.95×0.80 = 0.76 (standard gym/laboratory rubber mat).
// μ_roll=0.030: slightly higher rolling resistance than wood (rubber deforms under load).
};
constexpr int kNumGroundMaterials = (int)(sizeof(kGroundMaterials)/sizeof(kGroundMaterials[0]));

// Gravity environments
constexpr GravityEnv kGravities[] = {
    {"Earth",    9.81f},
    {"Moon",     1.62f},
    {"Mars",     3.72f},
    {"Jupiter", 24.79f},
    {"Neptune", 11.15f},
    {"Pluto",    0.62f},
    {"Sun",    274.00f},
    {"Zero-G",   0.00f},
};
constexpr int kNumGravities = (int)(sizeof(kGravities)/sizeof(kGravities[0]));

// Atmosphere types — name, ρ kg/m³, μ Pa·s
constexpr AtmosphereType kAtmospheres[] = {
    {"Vacuum",            0.000f,  0.0f},
    {"Mars  (CO2 thin)",  0.020f,  1.4e-5f},
    {"Earth (sea level)", 1.225f,  1.81e-5f},
    {"Venus (CO2 dense)", 65.00f,  3.5e-5f},
    {"Water",          1000.00f,  1.0e-3f},
};
constexpr int kNumAtmospheres = (int)(sizeof(kAtmospheres)/sizeof(kAtmospheres[0]));
