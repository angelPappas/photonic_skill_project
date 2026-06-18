import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web
import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
json_path = script_dir / "n_core.json"

# --------------------------------------------------------------------
# Simulation Parameters
# --------------------------------------------------------------------
wl_c = 1.55  # Central wavelength.
wl_bw = 0.100  # Wavelength bandwidth.
wl_n = 101  # Number of wavelength points to compute the output data.

n_sin = 2.0
n_sio2 = 1.444
n_si = 3.48
NA = 0.41
n_core = np.sqrt(NA**2 + n_sio2**2)

mat_sio2 = td.Medium(permittivity=n_sio2**2)  # Box and cladding material.

# Spot size of the gaussian mode launched by the lensed fiber at the taper tip.
spot_size = 3.2
core_diameter = 2.4
fiber_length = 8


# Wavelength and frequency values.
wl_range = np.linspace(wl_c - wl_bw / 2, wl_c + wl_bw / 2, wl_n)
freq_c = td.C_0 / wl_c
freq_range = td.C_0 / wl_range
freq_width = 0.5 * (np.max(freq_range) - np.min(freq_range))
run_time = 30 / freq_width

# --------------------------------------------------------------------
# Function for simulation definition
# --------------------------------------------------------------------
def get_fiber_simulation(source_x: float=2.0, fiber_length: float=fiber_length, n_core_test: float=n_core):

    fiber_core_medium = td.Medium(permittivity=n_core_test**2)

    core_fiber = td.Structure(
        geometry=td.Cylinder(
            axis=0,
            radius=core_diameter/2,
            length=fiber_length + 2.0,    # Adding 2um so that structure extends past the PMLs
            center=(fiber_length/2, 0, 0),
        ),
        medium=fiber_core_medium,
        name="fiber_core"
    )

    mode_spec = td.ModeSpec(num_modes=1, target_neff=n_core)
    fiber_mode = td.ModeSource(
        center=(source_x, 0, 0),
        size=(0, 14, 14),
        source_time=td.GaussianPulse(freq0=freq_c, fwidth=freq_width),
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
    )

    sim_fiber_test = td.Simulation(
        center=(fiber_length/2, 0, 0),
        size=(fiber_length, 16, 16),   # generous transverse extent
        medium=mat_sio2,               # background = fiber cladding
        structures=[core_fiber],  # just the core
        sources=[fiber_mode],          # ModeSource as currently defined
        monitors=[
            td.ModeMonitor(            # check power transmission and mode profile
                center=(fiber_length - 1, 0, 0),
                size=(0, 1e3, 1e3),
                freqs=[freq_c],
                mode_spec=td.ModeSpec(num_modes=1, target_neff=n_core_test),
                name="fiber_mode_out",
            ),
            td.FieldMonitor(           # visualize the cross-section
                center=(fiber_length -1, 0, 0),
                size=(0, 1e3, 1e3),
                freqs=[freq_c],
                name="fiber_cross_section",
            ),
        ],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=wl_c),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        run_time=run_time,
        symmetry=(0, -1, 1),
    )

    return sim_fiber_test

# -------------------------------------------------------------------------
# Auxilary functions
# -------------------------------------------------------------------------
# --- Find 1/e field radius (= 1/e² intensity radius = MFD/2) ---
def find_1_over_e_radius(coords, profile):
    peak = np.max(profile)
    threshold = peak / np.e   # 1/e field amplitude = 1/e² intensity
    # find where profile crosses threshold on both sides of peak
    peak_idx = np.argmax(profile)
    # left side
    left = np.where(profile[:peak_idx] < threshold)[0]
    left_r = coords[peak_idx] - coords[left[-1]] if len(left) > 0 else None
    # right side
    right = np.where(profile[peak_idx:] < threshold)[0]
    right_r = coords[peak_idx + right[0]] - coords[peak_idx] if len(right) > 0 else None
    return left_r, right_r


if __name__ == "__main__":
    # Adjust the value of the tested refractive index for the fiber's core
    n_core_test = 0.9922 * n_core

    sim_fiber = get_fiber_simulation(n_core_test=n_core_test, source_x=fiber_length-1.0-wl_c)

    sim_fiber_data = web.run(sim_fiber)

    # Extract Ey field at freq_c
    ey = sim_fiber_data["fiber_cross_section"].field_components["Ey"]
    ey_at_freq = np.abs(ey.sel(f=freq_c, method="nearest")).squeeze()

    # ey_at_freq has dims (y, z) — get coordinates
    y = ey_at_freq.coords["y"].values
    z = ey_at_freq.coords["z"].values
    ey_vals = ey_at_freq.values

    # --- 1D slice through center (z=0) to get y-profile ---
    z_center_idx = np.argmin(np.abs(z))
    y_profile = ey_vals[:, z_center_idx]

    # --- 1D slice through center (y=0) to get z-profile ---
    y_center_idx = np.argmin(np.abs(y))
    z_profile = ey_vals[y_center_idx, :]

    y_left, y_right = find_1_over_e_radius(y, y_profile)
    z_left, z_right = find_1_over_e_radius(z, z_profile)

    print(f"y-direction: left radius={y_left:.3f} µm, right radius={y_right:.3f} µm")
    print(f"y MFD ≈ {y_left + y_right:.3f} µm  (target: 3.2 µm)")
    print(f"z-direction: left radius={z_left:.3f} µm, right radius={z_right:.3f} µm")
    print(f"z MFD ≈ {z_left + z_right:.3f} µm  (target: 3.2 µm)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 2D map
    im = axes[0].pcolormesh(y, z, ey_vals.T, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("y (µm)")
    axes[0].set_ylabel("z (µm)")
    axes[0].set_title("|Ey| 2D cross-section")
    axes[0].set_aspect("equal")

    # y profile
    axes[1].plot(y, y_profile / y_profile.max())
    axes[1].axhline(1/np.e, color="red", linestyle="--", label="1/e threshold")
    if y_left and y_right:
        axes[1].axvline(-y_left, color="gray", linestyle=":")
        axes[1].axvline(y_right, color="gray", linestyle=":", label=f"MFD≈{y_left+y_right:.2f}µm")
    axes[1].set_xlabel("y (µm)")
    axes[1].set_ylabel("Normalised |Ey|")
    axes[1].set_title("y-profile (z=0)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # z profile
    axes[2].plot(z, z_profile / z_profile.max())
    axes[2].axhline(1/np.e, color="red", linestyle="--", label="1/e threshold")
    if z_left and z_right:
        axes[2].axvline(-z_left, color="gray", linestyle=":")
        axes[2].axvline(z_right, color="gray", linestyle=":", label=f"MFD≈{z_left+z_right:.2f}µm")
    axes[2].set_xlabel("z (µm)")
    axes[2].set_ylabel("Normalised |Ey|")
    axes[2].set_title("z-profile (y=0)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Fiber mode cross section for {round(n_core_test,4)} core refractive index and SiO2 (1.444) cladding")

    plt.tight_layout()
    plt.show()

    if np.abs(y_left + y_right - spot_size)/spot_size < 0.016:
        n_core_dict = {
            "n_core": n_core_test
        }
        with open(json_path, "w") as f:
            json.dump(n_core_dict, f, indent=4)