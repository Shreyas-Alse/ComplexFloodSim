import streamlit as st
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------
# 1. Modular Model Definitions
# ----------------------------------------------------
DANGER_THRESHOLD = 2.0  # meters
RATE_WINDOW_S = 30      # trailing window (s) used to estimate the rising rate

@dataclass
class Square:
    x: int
    y: int
    h: float
    drain_cap: float
    soil_weight: float = 1.0  # Relative weight for uneven distribution
    # Surface material fractions (soil + concrete = 1.0)
    soil_pct: float = 0.5
    concrete_pct: float = 0.5
    
    # Soil saturation tracking (m of water absorbed)
    soil_water_absorbed: float = 0.0
    soil_max_absorption: float = 0.15  # Max 150mm absorption per unit of soil
    
    # Dynamic fluid state
    water_level: float = 0.0
    water_rate_of_change: float = 0.0  # net m/s

    @property
    def total_height(self) -> float:
        return self.h + self.water_level

    @property
    def time_to_danger(self) -> float | None:
        if self.water_level >= DANGER_THRESHOLD:
            return 0.0
        if self.water_rate_of_change > 1e-6:
            return (DANGER_THRESHOLD - self.water_level) / self.water_rate_of_change
        return None


@dataclass
class SimulationContext:
    rainfall_rate_m_s: float
    drain_efficiency: float
    soil_infiltration_rate_m_s: float
    max_transfer_fraction: float = 0.20


# ----------------------------------------------------
# 2. Modular Pipeline Stages
# ----------------------------------------------------
def handle_precipitation(grid: dict[tuple[int, int], Square], ctx: SimulationContext) -> dict[tuple[int, int], float]:
    return {coord: ctx.rainfall_rate_m_s for coord in grid}


def handle_drainage(grid: dict[tuple[int, int], Square], ctx: SimulationContext) -> dict[tuple[int, int], float]:
    deltas = {}
    for coord, sq in grid.items():
        drain_limit = sq.drain_cap * ctx.drain_efficiency
        drained = min(sq.water_level, drain_limit)
        deltas[coord] = -drained
    return deltas


def handle_soil_infiltration(grid: dict[tuple[int, int], Square], ctx: SimulationContext) -> dict[tuple[int, int], float]:
    deltas = {}
    for coord, sq in grid.items():
        if sq.soil_pct <= 0 or sq.water_level <= 0:
            deltas[coord] = 0.0
            continue

        max_storage = sq.soil_pct * sq.soil_max_absorption
        remaining_storage = max(0.0, max_storage - sq.soil_water_absorbed)

        infiltrate_potential = ctx.soil_infiltration_rate_m_s * sq.soil_pct
        actual_infiltrated = min(sq.water_level, infiltrate_potential, remaining_storage)

        sq.soil_water_absorbed += actual_infiltrated
        deltas[coord] = -actual_infiltrated
    return deltas


def handle_hydraulic_flow(grid: dict[tuple[int, int], Square], ctx: SimulationContext) -> dict[tuple[int, int], float]:
    deltas = {coord: 0.0 for coord in grid}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for (x, y), sq in grid.items():
        if sq.water_level <= 0:
            continue

        lower_neighbors = []
        total_head_diff = 0.0

        for dx, dy in directions:
            neighbor = grid.get((x + dx, y + dy))
            if neighbor:
                diff = sq.total_height - neighbor.total_height
                if diff > 0:
                    lower_neighbors.append((neighbor, diff))
                    total_head_diff += diff

        if lower_neighbors:
            available_to_give = sq.water_level * ctx.max_transfer_fraction
            for neighbor, diff in lower_neighbors:
                transfer = available_to_give * (diff / total_head_diff)
                transfer = min(transfer, diff / 2.0)
                deltas[(x, y)] -= transfer
                deltas[(neighbor.x, neighbor.y)] += transfer

    return deltas


SIMULATION_PIPELINE = [
    handle_precipitation,
    handle_soil_infiltration,
    handle_drainage,
    handle_hydraulic_flow,
]


def step(grid: dict[tuple[int, int], Square], ctx: SimulationContext):
    net_deltas = {coord: 0.0 for coord in grid}

    for stage in SIMULATION_PIPELINE:
        stage_deltas = stage(grid, ctx)
        for coord, delta in stage_deltas.items():
            net_deltas[coord] += delta

    for coord, sq in grid.items():
        prev_water = sq.water_level
        sq.water_level = max(0.0, sq.water_level + net_deltas[coord])
        sq.water_rate_of_change = sq.water_level - prev_water


# ---- Timing fix: rising rate from a trailing window, not the last 1-s step ----
# Water hops between cells in pulses, so a single 1-s difference is very noisy and
# made "time to danger" wildly wrong. A trailing average (>= RATE_WINDOW_S seconds,
# independent of the Advance slider) gives a stable rate.
def new_level_history() -> deque:
    return deque(maxlen=RATE_WINDOW_S + 1)


def record_levels(history: deque, sim_time: float, grid: dict[tuple[int, int], Square]):
    history.append((sim_time, np.array([sq.water_level for sq in grid.values()])))


def apply_windowed_rates(history: deque, grid: dict[tuple[int, int], Square]):
    if len(history) < 2:
        return
    t_old, lvl_old = history[0]
    t_new, lvl_new = history[-1]
    span = t_new - t_old
    if span <= 0:
        return
    rates = (lvl_new - lvl_old) / span
    for sq, r in zip(grid.values(), rates):
        sq.water_rate_of_change = float(r)


def update_soil_distribution(grid: dict[tuple[int, int], Square], target_avg_soil: float):
    if target_avg_soil <= 0.0:
        for sq in grid.values():
            sq.soil_pct = 0.0
            sq.concrete_pct = 1.0
        return
    if target_avg_soil >= 1.0:
        for sq in grid.values():
            sq.soil_pct = 1.0
            sq.concrete_pct = 0.0
        return

    # Use getattr to prevent crashes if session state holds stale instances
    weights = np.array([getattr(sq, "soil_weight", 1.0) for sq in grid.values()])
    n_cells = len(weights)
    target_sum = target_avg_soil * n_cells

    low, high = 0.0, 10.0
    for _ in range(30):
        mid = (low + high) / 2.0
        projected_sum = np.clip(weights * mid, 0.0, 1.0).sum()
        if projected_sum < target_sum:
            low = mid
        else:
            high = mid

    for sq in grid.values():
        w = getattr(sq, "soil_weight", 1.0)
        soil_fraction = float(np.clip(w * mid, 0.0, 1.0))
        sq.soil_pct = soil_fraction
        sq.concrete_pct = 1.0 - soil_fraction


# ----------------------------------------------------
# 3. State & Grid Initialization
# ----------------------------------------------------
st.set_page_config(page_title="Stack Underflow", layout="wide")
st.title("🚨 Stack Underflow")

GRID_SIZE = 10

def initialize_new_grid():
    np.random.seed(42)
    raw_weights = np.random.uniform(0.1, 1.9, size=(GRID_SIZE, GRID_SIZE))
    raw_weights /= raw_weights.mean()

    st.session_state.grid = {
        (x, y): Square(
            x=x,
            y=y,
            h=float(np.random.randint(0, 5)),
            drain_cap=float(np.random.randint(0, 3)),
            soil_weight=float(raw_weights[y, x]),
            water_level=0.0,
        )
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
    }
    st.session_state.sim_time = 0
    st.session_state.level_history = new_level_history()
    update_soil_distribution(st.session_state.grid, target_avg_soil=0.50)

# Check if grid doesn't exist OR has old object definitions without soil_weight
if "grid" not in st.session_state or any(not hasattr(sq, "soil_weight") for sq in st.session_state.grid.values()):
    initialize_new_grid()
if "level_history" not in st.session_state:
    st.session_state.level_history = new_level_history()

# ----------------------------------------------------
# 4. Sidebar Controls
# ----------------------------------------------------
with st.sidebar:
    st.header("Simulation Settings")

    rainfall_mm_hr = st.slider(
        "Rainfall Rate (mm/hr)",
        min_value=50.0,
        max_value=1000.0,
        value=350.0,
        step=25.0,
        help="Values > 200 mm/hr represent flash floods."
    )
    rainfall_rate_m_s = (rainfall_mm_hr / 1000.0) / 3600.0

    st.subheader("Surface Material Distribution")
    soil_pct_input = st.slider(
        "Average Soil Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Total soil percentage averaged across all squares with uneven distribution."
    )
    target_soil_fraction = soil_pct_input / 100.0
    update_soil_distribution(st.session_state.grid, target_avg_soil=target_soil_fraction)

    actual_avg_soil = np.mean([sq.soil_pct for sq in st.session_state.grid.values()]) * 100.0
    actual_avg_concrete = 100.0 - actual_avg_soil
    st.caption(f"Grid Average: **{actual_avg_soil:.1f}% Soil / {actual_avg_concrete:.1f}% Concrete**")

    soil_k_sat_mm_hr = st.slider(
        "Soil Infiltration Rate (mm/hr)",
        min_value=5.0,
        max_value=150.0,
        value=40.0,
        step=5.0,
        help="Hydraulic conductivity of the soil fraction."
    )
    soil_k_sat_m_s = (soil_k_sat_mm_hr / 1000.0) / 3600.0

    drain_efficiency = st.slider(
        "Drain Efficiency",
        min_value=0.000005,
        max_value=0.000100,
        value=0.000020,
        step=0.000005,
        format="%.6f",
    )

    advance_seconds = st.slider("Advance Time (Seconds)", min_value=5, max_value=120, value=30, step=5)

    sim_ctx = SimulationContext(
        rainfall_rate_m_s=rainfall_rate_m_s,
        drain_efficiency=drain_efficiency,
        soil_infiltration_rate_m_s=soil_k_sat_m_s,
    )

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("▶ Run Step(s)"):
            history = st.session_state.level_history
            if len(history) == 0:
                record_levels(history, st.session_state.sim_time, st.session_state.grid)
            for _ in range(advance_seconds):
                step(st.session_state.grid, sim_ctx)
                st.session_state.sim_time += 1
                record_levels(history, st.session_state.sim_time, st.session_state.grid)
            # Stable rising rate over the trailing window (fixes time-to-danger)
            apply_windowed_rates(history, st.session_state.grid)
            st.rerun()

    with c_btn2:
        if st.button("🔄 Reset"):
            for sq in st.session_state.grid.values():
                sq.water_level = 0.0
                sq.water_rate_of_change = 0.0
                sq.soil_water_absorbed = 0.0
            st.session_state.sim_time = 0
            st.session_state.level_history = new_level_history()
            st.rerun()

    st.markdown("---")
    st.metric("Total Elapsed Time", f"{st.session_state.sim_time} s")
    total_surface_water = sum(sq.water_level for sq in st.session_state.grid.values())
    total_infiltrated_water = sum(sq.soil_water_absorbed for sq in st.session_state.grid.values())
    st.metric("Surface Standing Water", f"{total_surface_water:.3f} m³")
    st.metric("Total Absorbed by Soil", f"{total_infiltrated_water:.3f} m³")

# ----------------------------------------------------
# 5. Danger KPI Dashboard
# ----------------------------------------------------
total_families = len(st.session_state.grid)
flooded_now = sum(1 for sq in st.session_state.grid.values() if sq.water_level >= DANGER_THRESHOLD)
threatened_families = sum(
    1 for sq in st.session_state.grid.values()
    if sq.water_level < DANGER_THRESHOLD and sq.time_to_danger is not None
)
safe_families = total_families - flooded_now - threatened_families

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Elapsed Time", f"{st.session_state.sim_time} s")
kpi2.metric("🚨 Flooded Now (≥2.0m)", f"{flooded_now} / {total_families}")
kpi3.metric("⚠️ Rising / At Risk", f"{threatened_families} families")
kpi4.metric("🛡️ Safe / Draining", f"{safe_families} families")

st.markdown("---")

# ----------------------------------------------------
# 6. Visual Heatmaps (2 Rows, 2 in each Row)
# ----------------------------------------------------
terrain_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
water_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
absorbed_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
time_to_danger_matrix = np.full((GRID_SIZE, GRID_SIZE), np.nan)
danger_labels = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

for (x, y), sq in st.session_state.grid.items():
    terrain_matrix[y, x] = sq.h
    water_matrix[y, x] = sq.water_level
    absorbed_matrix[y, x] = sq.soil_water_absorbed * 1000.0
    ttd = sq.time_to_danger

    if ttd is not None:
        time_to_danger_matrix[y, x] = ttd
        danger_labels[y][x] = "FLOOD" if ttd == 0.0 else f"{ttd:.0f}s"
    else:
        danger_labels[y][x] = "SAFE"

# Row 1: Terrain & Water Depth
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig_terrain = go.Figure(
        data=go.Heatmap(
            z=terrain_matrix,
            colorscale="YlOrBr",
            colorbar=dict(title="Elevation (m)"),
            text=[[f"H: {terrain_matrix[r, c]:.1f}m" for c in range(GRID_SIZE)] for r in range(GRID_SIZE)],
            texttemplate="%{text}",
            textfont={"size": 11},
        )
    )
    fig_terrain.update_layout(
        title="1. Terrain Elevation (m)",
        xaxis_title="X", yaxis_title="Y", yaxis_autorange="reversed", height=450
    )
    st.plotly_chart(fig_terrain, use_container_width=True)

with row1_col2:
    fig_water = go.Figure(
        data=go.Heatmap(
            z=water_matrix,
            colorscale="Blues",
            zmin=0.0,
            zmax=max(0.5, float(np.max(water_matrix))),
            colorbar=dict(title="Depth (m)"),
            text=[[f"{water_matrix[r, c]:.3f}m" for c in range(GRID_SIZE)] for r in range(GRID_SIZE)],
            texttemplate="%{text}",
            textfont={"size": 11},
        )
    )
    fig_water.update_layout(
        title="2. Surface Water Depth (m)",
        xaxis_title="X", yaxis_title="Y", yaxis_autorange="reversed", height=450
    )
    st.plotly_chart(fig_water, use_container_width=True)

# Row 2: Soil Absorption & Time to Danger
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    fig_soil = go.Figure(
        data=go.Heatmap(
            z=absorbed_matrix,
            colorscale="YlGn",
            colorbar=dict(title="Absorbed (mm)"),
            text=[[f"{absorbed_matrix[r, c]:.1f}mm" for c in range(GRID_SIZE)] for r in range(GRID_SIZE)],
            texttemplate="%{text}",
            textfont={"size": 11},
        )
    )
    fig_soil.update_layout(
        title="3. Soil Absorption (mm)",
        xaxis_title="X", yaxis_title="Y", yaxis_autorange="reversed", height=450
    )
    st.plotly_chart(fig_soil, use_container_width=True)

with row2_col2:
    capped_times = np.nan_to_num(time_to_danger_matrix, nan=600.0)
    fig_timer = go.Figure(
        data=go.Heatmap(
            z=capped_times,
            colorscale="Reds_r",
            zmin=0.0,
            zmax=600.0,
            colorbar=dict(title="Seconds"),
            text=danger_labels,
            texttemplate="%{text}",
            textfont={"size": 11},
        )
    )
    fig_timer.update_layout(
        title="4. Time to Danger (≤600s)",
        xaxis_title="X", yaxis_title="Y", yaxis_autorange="reversed", height=450
    )
    st.plotly_chart(fig_timer, use_container_width=True)

# ----------------------------------------------------
# 7. Family Risk Board Table
# ----------------------------------------------------
st.subheader("📋 Family Evacuation & Surface Profile")

records = []
for (x, y), sq in st.session_state.grid.items():
    ttd = sq.time_to_danger
    if sq.water_level >= DANGER_THRESHOLD:
        status, sort_key, ttd_display = "🚨 FLOODED", 0.0, "0 s (Immediate Evac)"
    elif ttd is not None:
        status, sort_key, ttd_display = "⚠️ ENDANGERED", ttd, f"{ttd:.1f} s"
    else:
        status, sort_key, ttd_display = "🛡️ Safe", 99999.0, "No threat / Draining"

    records.append({
        "Location": f"({x}, {y})",
        "Elevation (m)": round(sq.h, 2),
        "Soil / Concrete": f"{int(sq.soil_pct*100)}% / {int(sq.concrete_pct*100)}%",
        "Water Level (m)": round(sq.water_level, 3),
        "Soil Infiltrated (mm)": round(sq.soil_water_absorbed * 1000.0, 1),
        "Rising Rate (mm/s)": round(sq.water_rate_of_change * 1000.0, 3),
        "Time to Danger": ttd_display,
        "Status": status,
        "_sort_key": sort_key,
    })

df = pd.DataFrame(records).sort_values(by="_sort_key").drop(columns=["_sort_key"])
st.dataframe(df, use_container_width=True, hide_index=True)