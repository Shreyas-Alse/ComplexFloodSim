import copy
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. GLOBAL ENGINE PHYSICAL CONSTANTS & CORE SOLVER
# ============================================================

GRAVITY = 9.81
DX = 10.0  # Cell size (meters)
MANNING_N = 0.015
CFL = 0.35
MAX_DT = 0.10
MIN_DEPTH = 1.0e-4  # wet/dry threshold [m]; below this a cell is "dry" for velocity
DANGER_THRESHOLD = 2.0  # meters
TERRAIN_RELIEF = 1.0  # scales the random bed relief (1.0 = original 0-4 m bumps)

ENSEMBLE_SIZE = 20
OBS_NOISE_STD = 0.0001

# --- Ensemble / data-assimilation settings ---
ENKF_INFLATION = 1.05  # multiplicative anomaly inflation
ENKF_LOC_RADIUS = 3.0  # Gaspari-Cohn half-width in cells (support = 2x)
ENS_RAIN_SIGMA = 0.20  # log-normal spread of member rainfall factor
ENS_KS_SIGMA = 0.25  # log-normal spread of member Ks factor
ENS_MANNING_SIGMA = 0.20  # log-normal spread of member Manning n
# "Nature run" that generates the synthetic sensor (hidden from the model)
TRUTH_RAIN_FACTOR = 1.15
TRUTH_KS_FACTOR = 0.80
TRUTH_MANNING_FACTOR = 1.25


class HydroMesh:

    def __init__(self, grid_size=(10, 10), elevation_base=5.0, _build_ensemble=True):
        self.rows, self.cols = grid_size

        # Local RNG state (same terrain as before, but does NOT reseed the global RNG)
        rs = np.random.RandomState(42)
        self.z = rs.randint(0, 5, size=grid_size).astype(float) * TERRAIN_RELIEF
        for r in range(self.rows):
            for c in range(self.cols):
                self.z[r, c] += (self.rows - r) * 0.05 + (self.cols - c) * 0.02
        # Make the drain cell (bottom-right) the lowest point so the outlet is physical
        self.z[-1, -1] = min(self.z[-1, -1], float(np.min(self.z)))

        # Conservative state variables: depth h [m], discharge hu [m^2/s], hv [m^2/s]
        self.h = np.zeros(grid_size, dtype=float)
        self.hu = np.zeros(grid_size, dtype=float)
        self.hv = np.zeros(grid_size, dtype=float)
        self.water_rate_of_change = np.zeros(grid_size, dtype=float)

        # Soil & surface properties
        raw_weights = rs.uniform(0.1, 1.9, size=grid_size)
        self.soil_weight = raw_weights / raw_weights.mean()
        self.soil_pct = np.full(grid_size, 0.5, dtype=float)
        self.concrete_pct = np.full(grid_size, 0.5, dtype=float)

        # Green-Ampt infiltration parameters
        self.ks_base = 40.0 / 1000.0 / 3600.0  # m/s (40 mm/hr default)
        self.psi_dtheta = 150.0 * 0.20 / 1000.0  # 30 mm suction parameter
        # F_cum = cumulative infiltration per unit SOIL area [m]
        self.F_cum = np.full(grid_size, 0.001, dtype=float)
        # soil_water_absorbed = infiltrated depth averaged over the whole cell [m]
        self.soil_water_absorbed = np.zeros(grid_size, dtype=float)

        # Orifice drain boundary conditions (drain cell = bottom-right)
        self.pipe_hgl = float(self.z[-1, -1]) - 0.5
        self.orifice_area = 0.45
        self.cd = 0.62

        # Per-run parameters (perturbed for ensemble members)
        self.rain_factor = 1.0
        self.ks_factor = 1.0
        self.manning_n = MANNING_N

        # Accumulation diagnostics
        self.total_rain_depth = 0.0
        self.total_infiltration_depth = 0.0
        self.total_drain_volume = 0.0
        self.clipped_volume = 0.0  # water created by negative-depth clipping [m^3]
        self.da_volume = 0.0  # net water added by data assimilation [m^3]

        # Data assimilation: propagated ensemble + hidden "nature run" for the sensor
        self.rng = np.random.default_rng(2024)
        self.ensemble = []
        self.truth = None
        if _build_ensemble:
            self._build_ensemble()

    # ---------------- ensemble construction ----------------
    def _clone(self):
        m = copy.copy(self)
        m.h = self.h.copy()
        m.hu = self.hu.copy()
        m.hv = self.hv.copy()
        m.F_cum = self.F_cum.copy()
        m.soil_water_absorbed = self.soil_water_absorbed.copy()
        m.water_rate_of_change = self.water_rate_of_change.copy()
        m.ensemble = []
        m.truth = None
        return m

    def _build_ensemble(self):
        self.ensemble = []
        for _ in range(ENSEMBLE_SIZE):
            m = self._clone()
            m.rain_factor = float(np.exp(self.rng.normal(0.0, ENS_RAIN_SIGMA)))
            m.ks_factor = float(np.exp(self.rng.normal(0.0, ENS_KS_SIGMA)))
            m.manning_n = MANNING_N * float(np.exp(self.rng.normal(0.0, ENS_MANNING_SIGMA)))
            self.ensemble.append(m)
        t = self._clone()
        t.rain_factor = TRUTH_RAIN_FACTOR
        t.ks_factor = TRUTH_KS_FACTOR
        t.manning_n = MANNING_N * TRUTH_MANNING_FACTOR
        self.truth = t

    # ---------------- derived fields ----------------
    @property
    def u(self):
        return np.divide(
            self.hu, self.h, out=np.zeros_like(self.h), where=self.h > MIN_DEPTH
        )

    @property
    def v(self):
        return np.divide(
            self.hv, self.h, out=np.zeros_like(self.h), where=self.h > MIN_DEPTH
        )

    @property
    def total_height(self):
        return self.z + self.h

    def validate(self):
        arrays = {
            "h": self.h,
            "hu": self.hu,
            "hv": self.hv,
            "F_cum": self.F_cum,
        }
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"Mesh array {name} contains NaN/Inf.")
        # Clip negatives, but account for the water this creates
        neg = np.minimum(self.h, 0.0)
        if neg.any():
            self.clipped_volume += float(-neg.sum()) * DX * DX
            self.h = self.h - neg

    def mass_balance_residual(self):
        """stored - (rain - infiltrated - drained + assimilation + clipping) [m^3].
        Should be ~0 (round-off) if every source/sink is accounted for."""
        area = DX * DX
        rain = self.total_rain_depth * self.rows * self.cols * area
        infil = float(np.sum(self.soil_water_absorbed)) * area
        stored = float(np.sum(self.h)) * area
        expected = (
            rain - infil - self.total_drain_volume + self.da_volume + self.clipped_volume
        )
        return stored - expected


def update_soil_distribution(mesh: HydroMesh, target_avg_soil: float):
    if target_avg_soil <= 0.0:
        mesh.soil_pct[:] = 0.0
        mesh.concrete_pct[:] = 1.0
        return
    if target_avg_soil >= 1.0:
        mesh.soil_pct[:] = 1.0
        mesh.concrete_pct[:] = 0.0
        return

    weights = mesh.soil_weight
    target_sum = target_avg_soil * weights.size
    low = 0.0
    high = 1.0 / max(float(weights.min()), 1.0e-6) + 1.0  # guarantees full saturation

    for _ in range(60):
        mid = (low + high) / 2.0
        projected_sum = np.clip(weights * mid, 0.0, 1.0).sum()
        if projected_sum < target_sum:
            low = mid
        else:
            high = mid

    mesh.soil_pct = np.clip(weights * high, 0.0, 1.0)
    mesh.concrete_pct = 1.0 - mesh.soil_pct


def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(num), where=den > MIN_DEPTH)


def _hr_flux(hL, qnL, qtL, zL, hR, qnR, qtR, zR):
    """Rusanov flux with Audusse hydrostatic reconstruction at one family of
    interfaces. qn = momentum normal to the interface, qt = tangential momentum.

    Returns (mass flux, normal-momentum flux, tangential-momentum flux,
             pressure correction for left cell, pressure correction for right cell).
    The correction terms replace the centred bed-slope source, making the scheme
    well-balanced (lake-at-rest preserved) and preventing uphill mass transfer."""
    unL, utL = _safe_div(qnL, hL), _safe_div(qtL, hL)
    unR, utR = _safe_div(qnR, hR), _safe_div(qtR, hR)

    z_face = np.maximum(zL, zR)
    hLs = np.maximum(0.0, hL + zL - z_face)
    hRs = np.maximum(0.0, hR + zR - z_face)

    a = np.maximum(
        np.abs(unL) + np.sqrt(GRAVITY * hLs), np.abs(unR) + np.sqrt(GRAVITY * hRs)
    )

    qnLs, qnRs = hLs * unL, hRs * unR
    qtLs, qtRs = hLs * utL, hRs * utR

    FL_m, FR_m = qnLs, qnRs
    FL_n = qnLs * unL + 0.5 * GRAVITY * hLs**2
    FR_n = qnRs * unR + 0.5 * GRAVITY * hRs**2
    FL_t, FR_t = qnLs * utL, qnRs * utR

    F_m = 0.5 * (FL_m + FR_m) - 0.5 * a * (hRs - hLs)
    F_n = 0.5 * (FL_n + FR_n) - 0.5 * a * (qnRs - qnLs)
    F_t = 0.5 * (FL_t + FR_t) - 0.5 * a * (qtRs - qtLs)

    corr_L = 0.5 * GRAVITY * (hL**2 - hLs**2)
    corr_R = 0.5 * GRAVITY * (hR**2 - hRs**2)
    return F_m, F_n, F_t, corr_L, corr_R


def stable_dt(mesh):
    c = np.sqrt(GRAVITY * mesh.h)
    wave_speed_x = np.abs(mesh.u) + c
    wave_speed_y = np.abs(mesh.v) + c
    max_speed = max(
        float(np.max(wave_speed_x)), float(np.max(wave_speed_y)), 1.0e-12
    )
    return min(MAX_DT, CFL * DX / max_speed)


def solve_shallow_water_step(mesh, dt):
    h, hu, hv, z = mesh.h, mesh.hu, mesh.hv, mesh.z
    k = dt / DX

    dh = np.zeros_like(h)
    dhu = np.zeros_like(h)
    dhv = np.zeros_like(h)

    # ---- x-direction interfaces (normal = hu, tangential = hv) ----
    Fm, Fn, Ft, cL, cR = _hr_flux(
        h[:, :-1], hu[:, :-1], hv[:, :-1], z[:, :-1],
        h[:, 1:], hu[:, 1:], hv[:, 1:], z[:, 1:],
    )
    dh[:, :-1] -= k * Fm
    dh[:, 1:] += k * Fm
    dhu[:, :-1] -= k * (Fn + cL)
    dhu[:, 1:] += k * (Fn + cR)
    dhv[:, :-1] -= k * Ft
    dhv[:, 1:] += k * Ft

    # ---- y-direction interfaces (normal = hv, tangential = hu) ----
    Gm, Gn, Gt, cB, cT = _hr_flux(
        h[:-1, :], hv[:-1, :], hu[:-1, :], z[:-1, :],
        h[1:, :], hv[1:, :], hu[1:, :], z[1:, :],
    )
    dh[:-1, :] -= k * Gm
    dh[1:, :] += k * Gm
    dhv[:-1, :] -= k * (Gn + cB)
    dhv[1:, :] += k * (Gn + cT)
    dhu[:-1, :] -= k * Gt
    dhu[1:, :] += k * Gt

    # ---- closed (reflective wall) boundaries: F = (0, 1/2 g h^2, 0) ----
    dhu[:, 0] += k * 0.5 * GRAVITY * h[:, 0] ** 2
    dhu[:, -1] -= k * 0.5 * GRAVITY * h[:, -1] ** 2
    dhv[0, :] += k * 0.5 * GRAVITY * h[0, :] ** 2
    dhv[-1, :] -= k * 0.5 * GRAVITY * h[-1, :] ** 2

    h_new = h + dh
    hu_new = hu + dhu
    hv_new = hv + dhv

    dry = h_new <= MIN_DEPTH
    hu_new = np.where(dry, 0.0, hu_new)
    hv_new = np.where(dry, 0.0, hv_new)

    u_new = _safe_div(hu_new, h_new)
    v_new = _safe_div(hv_new, h_new)
    speed = np.sqrt(u_new**2 + v_new**2)

    # Semi-implicit Manning friction
    friction_factor = 1.0 + (
        GRAVITY
        * mesh.manning_n**2
        * speed
        * dt
        / np.maximum(h_new, MIN_DEPTH) ** (4.0 / 3.0)
    )

    mesh.h = h_new  # negatives (round-off) are clipped & accounted for in validate()
    mesh.hu = hu_new / friction_factor
    mesh.hv = hv_new / friction_factor
    mesh.validate()


def apply_rainfall_and_infiltration(mesh, rain_mm_hr, ks_m_s, dt):
    rain_rate = rain_mm_hr / 1000.0 / 3600.0
    soil = mesh.soil_pct
    soil_safe = np.maximum(soil, 1.0e-9)

    # Green-Ampt capacity per unit soil area, scaled to the cell by soil fraction
    F_soil = np.maximum(mesh.F_cum, 1.0e-5)
    capacity = ks_m_s * (1.0 + mesh.psi_dtheta / F_soil) * soil
    # Water available to infiltrate: rain landing on soil + ponded water over soil
    supply = (rain_rate + mesh.h / dt) * soil
    infiltration = np.minimum(capacity, supply)

    infiltrated_depth = infiltration * dt  # cell-averaged depth [m]
    mesh.F_cum += infiltrated_depth / soil_safe  # per unit soil area
    mesh.soil_water_absorbed += infiltrated_depth

    old_h = mesh.h.copy()
    new_h = old_h + rain_rate * dt - infiltrated_depth
    new_h = np.where(new_h < 0.0, 0.0, new_h)  # only round-off can reach here

    # Rain has no horizontal momentum: keep hu when depth grows, dilute only if water is lost
    momentum_factor = np.where(
        old_h > MIN_DEPTH, np.minimum(1.0, new_h / np.maximum(old_h, MIN_DEPTH)), 0.0
    )
    mesh.hu *= momentum_factor
    mesh.hv *= momentum_factor
    mesh.h = new_h

    mesh.total_rain_depth += rain_rate * dt
    mesh.total_infiltration_depth += float(np.mean(infiltrated_depth))
    mesh.validate()


def apply_orifice_drain(mesh, tailwater_elev, dt):
    """Returns the ACTUAL discharge [m^3/s] removed this sub-step."""
    r, c = mesh.rows - 1, mesh.cols - 1
    h = mesh.h[r, c]

    if h <= MIN_DEPTH:
        return 0.0

    surface_elev = mesh.z[r, c] + h
    head = surface_elev - max(mesh.pipe_hgl, tailwater_elev)

    if head <= 0.0:
        return 0.0

    discharge = mesh.cd * mesh.orifice_area * math.sqrt(2.0 * GRAVITY * head)
    requested_depth_drop = discharge * dt / (DX * DX)
    actual_depth_drop = min(requested_depth_drop, h)

    old_h = h
    new_h = old_h - actual_depth_drop
    momentum_factor = new_h / old_h if old_h > MIN_DEPTH else 0.0

    mesh.h[r, c] = new_h
    mesh.hu[r, c] *= momentum_factor
    mesh.hv[r, c] *= momentum_factor
    mesh.total_drain_volume += actual_depth_drop * DX * DX

    mesh.validate()
    return actual_depth_drop * DX * DX / dt


def advance_mesh(mesh, duration, rain_mm_hr, ks_m_s, tailwater_elev):
    """Adaptive CFL sub-stepping for one mesh over `duration` seconds.
    Uses the mesh's own rain/Ks/Manning factors (1.0 for the control run).
    Returns (elapsed seconds, mean actual drain discharge over the window)."""
    elapsed = 0.0
    drained_before = mesh.total_drain_volume
    while elapsed < duration - 1.0e-9:
        dt = min(stable_dt(mesh), duration - elapsed)
        solve_shallow_water_step(mesh, dt)
        apply_rainfall_and_infiltration(
            mesh, rain_mm_hr * mesh.rain_factor, ks_m_s * mesh.ks_factor, dt
        )
        apply_orifice_drain(mesh, tailwater_elev, dt)
        elapsed += dt
    q_mean = (mesh.total_drain_volume - drained_before) / max(elapsed, 1.0e-9)
    return elapsed, q_mean


def advance_ensemble(mesh, duration, rain_mm_hr, ks_m_s, tailwater_elev):
    """Propagate every ensemble member (and the hidden nature run) through the SAME
    solver with their own perturbed rainfall / infiltration / roughness."""
    for m in [*mesh.ensemble, mesh.truth]:
        m.soil_pct = mesh.soil_pct
        m.concrete_pct = mesh.concrete_pct
        advance_mesh(m, duration, rain_mm_hr, ks_m_s, tailwater_elev)


def _zero_dry_momentum(m):
    dry = m.h <= MIN_DEPTH
    m.hu = np.where(dry, 0.0, m.hu)
    m.hv = np.where(dry, 0.0, m.hv)


def _gaspari_cohn(dist, c):
    r = dist / c
    out = np.zeros_like(r)
    m1 = r <= 1.0
    m2 = (r > 1.0) & (r <= 2.0)
    r1 = r[m1]
    r2 = r[m2]
    out[m1] = -0.25 * r1**5 + 0.5 * r1**4 + 0.625 * r1**3 - (5.0 / 3.0) * r1**2 + 1.0
    out[m2] = (
        (1.0 / 12.0) * r2**5 - 0.5 * r2**4 + 0.625 * r2**3
        + (5.0 / 3.0) * r2**2 - 5.0 * r2 + 4.0 - (2.0 / 3.0) / r2
    )
    return out


def run_enkf_assimilation(mesh, sensor_row, sensor_col):
    """Stochastic, localized EnKF using the propagated ensemble.

    - Observation = hidden nature-run depth at the sensor + noise.
    - Each member assimilates its own perturbed observation.
    - Gains come from ensemble cross-covariances of (h, hu, hv) with the sensor,
      tapered by a Gaspari-Cohn function so distant cells are not touched.
    - The control run (the state shown on the dashboard) receives the same gain.
    Returns (forecast_error, analysis_error) at the sensor for the control run."""
    rng = mesh.rng
    members = mesh.ensemble
    n = len(members)
    sr, sc = sensor_row, sensor_col

    obs = float(mesh.truth.h[sr, sc] + rng.normal(0.0, OBS_NOISE_STD))
    forecast_error = abs(float(mesh.h[sr, sc]) - obs)

    X = np.stack(
        [np.stack((m.h, m.hu, m.hv), axis=0) for m in members], axis=0
    )  # (N, 3, R, C)
    x_mean = X.mean(axis=0)
    X = x_mean + ENKF_INFLATION * (X - x_mean)
    X[:, 0] = np.maximum(X[:, 0], 0.0)

    y = X[:, 0, sr, sc]
    y_anom = y - y.mean()
    x_anom = X - X.mean(axis=0)
    cov_xy = np.einsum("n,nkrc->krc", y_anom, x_anom) / (n - 1)
    var_y = float(y_anom @ y_anom) / (n - 1)

    rr, cc = np.indices((mesh.rows, mesh.cols))
    loc = _gaspari_cohn(np.hypot(rr - sr, cc - sc).astype(float), ENKF_LOC_RADIUS)

    gain = cov_xy / (var_y + OBS_NOISE_STD**2 + 1.0e-12) * loc[None]

    obs_pert = obs + rng.normal(0.0, OBS_NOISE_STD, size=n)
    X_a = X + gain[None] * (obs_pert - y)[:, None, None, None]

    for i, m in enumerate(members):
        m.h = np.maximum(X_a[i, 0], 0.0)
        m.hu = X_a[i, 1]
        m.hv = X_a[i, 2]
        _zero_dry_momentum(m)

    # Control run analysis (same gain, its own innovation)
    inc = gain * (obs - float(mesh.h[sr, sc]))
    mesh.da_volume += float(inc[0].sum()) * DX * DX
    mesh.h = mesh.h + inc[0]
    mesh.hu = mesh.hu + inc[1]
    mesh.hv = mesh.hv + inc[2]
    mesh.validate()
    _zero_dry_momentum(mesh)

    analysis_error = abs(float(mesh.h[sr, sc]) - obs)
    return forecast_error, analysis_error


# ============================================================
# 2. STATE & MESH INITIALIZATION
# ============================================================
st.set_page_config(page_title="Stack Underflow SWE Engine", layout="wide")
st.title("🚨 Stack Underflow: 2D Shallow Water Hydro Engine")

GRID_SIZE = 10

if "hydro_mesh" not in st.session_state:
    st.session_state.hydro_mesh = HydroMesh(grid_size=(GRID_SIZE, GRID_SIZE))
    st.session_state.sim_time = 0.0
    st.session_state.last_fc_err = 0.0
    st.session_state.last_an_err = 0.0
    st.session_state.last_drain_q = 0.0
    update_soil_distribution(st.session_state.hydro_mesh, target_avg_soil=0.50)

mesh = st.session_state.hydro_mesh

# ============================================================
# 3. SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Simulation Physical Parameters")

    rainfall_mm_hr = st.slider(
        "Rainfall Rate (mm/hr)",
        min_value=0.0,
        max_value=1000.0,
        value=350.0,
        step=25.0,
    )

    st.subheader("Surface Material Distribution")
    soil_pct_input = st.slider(
        "Average Soil Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
    )
    update_soil_distribution(mesh, target_avg_soil=soil_pct_input / 100.0)

    soil_k_sat_mm_hr = st.slider(
        "Soil Hydraulic Cond. Ks (mm/hr)",
        min_value=5.0,
        max_value=150.0,
        value=40.0,
        step=5.0,
    )
    ks_m_s = (soil_k_sat_mm_hr / 1000.0) / 3600.0

    tailwater_elev = st.slider(
        "Drain Tailwater Elevation (m)",
        min_value=float(np.min(mesh.z) - 2.0),
        max_value=float(np.max(mesh.z) + 1.0),
        value=float(np.min(mesh.z) - 1.0),
        step=0.1,
    )

    advance_target_sec = st.slider(
        "Advance Integration Window (Seconds)",
        min_value=1.0,
        max_value=300.0,
        value=5.0,
        step=1.0,
    )

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("▶ Run Hydro Step(s)"):
            last_h = mesh.h.copy()

            # Control run: adaptive sub-stepping obeying CFL
            elapsed_in_batch, q_drain = advance_mesh(
                mesh, advance_target_sec, rainfall_mm_hr, ks_m_s, tailwater_elev
            )

            # Propagate the perturbed ensemble + hidden nature run through the solver
            advance_ensemble(
                mesh, advance_target_sec, rainfall_mm_hr, ks_m_s, tailwater_elev
            )

            # Compute net rate of change per second
            mesh.water_rate_of_change = (mesh.h - last_h) / max(
                elapsed_in_batch, 1e-6
            )

            # EnKF assimilation on bottom-right sensor
            fc_err, an_err = run_enkf_assimilation(
                mesh, mesh.rows - 1, mesh.cols - 1
            )

            st.session_state.sim_time += elapsed_in_batch
            st.session_state.last_fc_err = fc_err
            st.session_state.last_an_err = an_err
            st.session_state.last_drain_q = q_drain
            st.rerun()

    with c_btn2:
        if st.button("🔄 Reset Engine"):
            st.session_state.hydro_mesh = HydroMesh(
                grid_size=(GRID_SIZE, GRID_SIZE)
            )
            st.session_state.sim_time = 0.0
            st.session_state.last_fc_err = 0.0
            st.session_state.last_an_err = 0.0
            st.session_state.last_drain_q = 0.0
            update_soil_distribution(
                st.session_state.hydro_mesh,
                target_avg_soil=soil_pct_input / 100.0,
            )
            st.rerun()

    st.markdown("---")
    st.metric("Total Elapsed Time", f"{st.session_state.sim_time:.2f} s")
    total_surface_water = float(np.sum(mesh.h) * (DX * DX))
    total_infiltrated = float(np.sum(mesh.soil_water_absorbed) * (DX * DX))
    st.metric("Surface Standing Volume", f"{total_surface_water:.2f} m³")
    st.metric("Infiltrated Soil Volume", f"{total_infiltrated:.2f} m³")
    st.metric(
        "Drain Discharge Outflow", f"{st.session_state.last_drain_q:.4f} m³/s"
    )

# ============================================================
# 4. KPI DASHBOARD & ENKF RESIDUALS
# ============================================================
total_cells = mesh.rows * mesh.cols
flooded_cells = int(np.sum(mesh.h >= DANGER_THRESHOLD))
at_risk_cells = int(
    np.sum((mesh.h < DANGER_THRESHOLD) & (mesh.water_rate_of_change > 1e-5))
)
safe_cells = total_cells - flooded_cells - at_risk_cells

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Simulation Time", f"{st.session_state.sim_time:.1f} s")
kpi2.metric(f"🚨 Flooded (≥{DANGER_THRESHOLD}m)", f"{flooded_cells} / {total_cells}")
kpi3.metric("⚠️ Water Rising", f"{at_risk_cells} cells")
kpi4.metric(
    "Sensor Forecast Error", f"{st.session_state.last_fc_err * 1000.0:.3f} mm"
)
kpi5.metric(
    "EnKF Analysis Error", f"{st.session_state.last_an_err * 1000.0:.3f} mm"
)

st.markdown("---")

# ============================================================
# 5. DUAL-ROW VISUAL HEATMAPS
# ============================================================
velocity_mag = np.sqrt(mesh.u**2 + mesh.v**2)

time_to_danger = np.full((GRID_SIZE, GRID_SIZE), np.nan)
danger_labels = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

for r in range(mesh.rows):
    for c in range(mesh.cols):
        h = mesh.h[r, c]
        dh = mesh.water_rate_of_change[r, c]
        if h >= DANGER_THRESHOLD:
            time_to_danger[r, c] = 0.0
            danger_labels[r][c] = "FLOOD"
        elif dh > 1e-5:
            ttd = (DANGER_THRESHOLD - h) / dh
            time_to_danger[r, c] = ttd
            danger_labels[r][c] = f"{ttd:.0f}s"
        else:
            danger_labels[r][c] = "SAFE"

# Row 1: Bed Elevation & SWE Water Depth
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig_terrain = go.Figure(
        data=go.Heatmap(
            z=mesh.z,
            colorscale="YlOrBr",
            colorbar=dict(title="Elevation (m)"),
            text=[
                [f"Z: {mesh.z[r, c]:.2f}m" for c in range(GRID_SIZE)]
                for r in range(GRID_SIZE)
            ],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig_terrain.update_layout(
        title="1. Bed Elevation & Terrain Gradients (m)",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange="reversed",
        height=420,
    )
    st.plotly_chart(fig_terrain, use_container_width=True)

with row1_col2:
    fig_water = go.Figure(
        data=go.Heatmap(
            z=mesh.h,
            colorscale="Blues",
            zmin=0.0,
            zmax=max(0.5, float(np.max(mesh.h))),
            colorbar=dict(title="Depth (m)"),
            text=[
                [f"{mesh.h[r, c]:.3f}m" for c in range(GRID_SIZE)]
                for r in range(GRID_SIZE)
            ],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig_water.update_layout(
        title="2. SWE Solved Water Depth (m)",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange="reversed",
        height=420,
    )
    st.plotly_chart(fig_water, use_container_width=True)

# Row 2: Green-Ampt Cumulative Infiltration & Hydrodynamic Velocity Fields
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    absorbed_mm = mesh.soil_water_absorbed * 1000.0
    fig_soil = go.Figure(
        data=go.Heatmap(
            z=absorbed_mm,
            colorscale="YlGn",
            colorbar=dict(title="Absorbed (mm)"),
            text=[
                [
                    f"{absorbed_mm[r, c]:.1f}mm<br>({int(mesh.soil_pct[r, c]*100)}% S)"
                    for c in range(GRID_SIZE)
                ]
                for r in range(GRID_SIZE)
            ],
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    fig_soil.update_layout(
        title="3. Green-Ampt Infiltration & Soil Allocation (mm)",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange="reversed",
        height=420,
    )
    st.plotly_chart(fig_soil, use_container_width=True)

with row2_col2:
    fig_vel = go.Figure(
        data=go.Heatmap(
            z=velocity_mag,
            colorscale="Plasma",
            colorbar=dict(title="Velocity (m/s)"),
            text=[
                [f"{velocity_mag[r, c]:.3f} m/s" for c in range(GRID_SIZE)]
                for r in range(GRID_SIZE)
            ],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig_vel.update_layout(
        title="4. Flow Velocity Field (|u, v| in m/s)",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange="reversed",
        height=420,
    )
    st.plotly_chart(fig_vel, use_container_width=True)

# ============================================================
# 6. CELL-BY-CELL HYDRO PROFILE TABLE
# ============================================================
st.subheader("📋 Hydrodynamic State & Evacuation Risk Board")

records = []
for r in range(mesh.rows):
    for c in range(mesh.cols):
        h = mesh.h[r, c]
        dh = mesh.water_rate_of_change[r, c]
        vel = velocity_mag[r, c]

        if h >= DANGER_THRESHOLD:
            status, sort_key, ttd_str = (
                "🚨 FLOODED",
                0.0,
                "0.0 s (Immediate Evac)",
            )
        elif dh > 1e-5:
            ttd = (DANGER_THRESHOLD - h) / dh
            status, sort_key, ttd_str = "⚠️ RISING", ttd, f"{ttd:.1f} s"
        else:
            status, sort_key, ttd_str = (
                "🛡️ SAFE",
                99999.0,
                "Draining / No threat",
            )

        records.append(
            {
                "Cell": f"({r}, {c})",
                "Bed Elev (m)": round(mesh.z[r, c], 2),
                "Depth (m)": round(h, 4),
                "Velocity (m/s)": round(vel, 4),
                "Soil %": f"{int(mesh.soil_pct[r, c]*100)}%",
                "Infiltrated (mm)": round(mesh.soil_water_absorbed[r, c] * 1000.0, 2),
                "Rising Rate (mm/s)": round(dh * 1000.0, 3),
                "Time to Danger": ttd_str,
                "Status": status,
                "_sort_key": sort_key,
            }
        )

df = pd.DataFrame(records).sort_values(by="_sort_key").drop(columns=["_sort_key"])
st.dataframe(df, use_container_width=True, hide_index=True)