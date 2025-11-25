import meep as mp
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
from tqdm import tqdm


# === simulation parameters ===
wavelength = 1.55
fcen = 1/wavelength
df = 0.2*fcen
n_Si = 3.48  # silicon
n_SiO2 = 1.444  # SiO2

def mmi2x2(
        wg_length: float=3, wg_width: float = 0.4, taper_length: float=8, taper_width: float=0.6, mmi_width: float=2.55, mmi_length: float=30,
        THREE_D: bool=False):

    THREE_D = THREE_D

    resolution = 25    # pixels/µm

    t_SiO2 = 1.0
    t_Si = 0.22 if THREE_D else 20
    t_air = 0.78

    # geometry
    wg_width = wg_width
    wg_length = wg_length

    taper_length = taper_length
    taper_width = taper_width

    mmi_width = mmi_width
    mmi_length = mmi_length

    gap_between_ports = mmi_width/2
    
    # The y-coordinate of the center of the waveguides is at +- gap_between_ports/2
    # If I make the monitors too wide, they will overlap! This is why I choose them to be min(gap_between_ports, 2),
    # making sure they stay on their half of the simulation region (y>0 or y<0).
    monitor_width = wg_width + 0.2

    dpml = 1
    cell_thickness = (dpml + t_SiO2 + t_Si + t_air + dpml) if THREE_D else 0

    oxide = mp.Medium(epsilon=n_SiO2**2)

    # === simulation cell ===
    sx = wg_length + taper_length + mmi_length + wg_length + taper_length     # padding
    sy = dpml + mmi_width + dpml + 2    # adding one for um so that the pmls are not right next to the Si's multimode region
    cell = mp.Volume(
        size=mp.Vector3(sx, sy, cell_thickness),
        dims=3
    )

    pml_layers = [mp.PML(1.0)]

    # === geometry definition ===
    geometry = [
        # multimode region
        mp.Block(size=mp.Vector3(mmi_length, mmi_width, t_Si),
                center=mp.Vector3(0,0),
                material=mp.Medium(epsilon=n_Si**2)),

        # left waveguides
        mp.Block(size=mp.Vector3(wg_length, wg_width, t_Si),
                center=mp.Vector3(-mmi_length/2 - wg_length/2 - taper_length, gap_between_ports/2),
                material=mp.Medium(epsilon=n_Si**2)),
        
        mp.Prism(vertices=
                [mp.Vector3(-mmi_length/2 - taper_length, gap_between_ports/2-wg_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2, gap_between_ports/2 - taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2, gap_between_ports/2 + taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2 - taper_length, gap_between_ports/2+wg_width/2, -t_Si/2 if THREE_D else 0.0)],
                height=t_Si,
                material=mp.Medium(epsilon=n_Si**2)),

        mp.Block(size=mp.Vector3(wg_length, wg_width, t_Si),
                center=mp.Vector3(-mmi_length/2 - wg_length/2 - taper_length, -gap_between_ports/2),
                material=mp.Medium(epsilon=n_Si**2)),
        
        mp.Prism(vertices=
                [mp.Vector3(-mmi_length/2 - taper_length, -gap_between_ports/2-wg_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2, -gap_between_ports/2 - taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2, -gap_between_ports/2 + taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(-mmi_length/2 - taper_length, -gap_between_ports/2+wg_width/2, -t_Si/2 if THREE_D else 0.0)],
                height=t_Si,
                material=mp.Medium(epsilon=n_Si**2)),

        # right waveguides
        mp.Block(size=mp.Vector3(wg_length, wg_width, t_Si),
                center=mp.Vector3(mmi_length/2 + wg_length/2 + taper_length, gap_between_ports/2),
                material=mp.Medium(epsilon=n_Si**2)),
        
        mp.Prism(vertices=
                [mp.Vector3(mmi_length/2 + taper_length, gap_between_ports/2-wg_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2, gap_between_ports/2 - taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2, gap_between_ports/2 + taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2 + taper_length, gap_between_ports/2+wg_width/2, -t_Si/2 if THREE_D else 0.0)],
                height=t_Si,
                material=mp.Medium(epsilon=n_Si**2)),

        mp.Block(size=mp.Vector3(wg_length, wg_width, t_Si),
                center=mp.Vector3(mmi_length/2 + wg_length/2 + taper_length, -gap_between_ports/2),
                material=mp.Medium(epsilon=n_Si**2)),
        
        mp.Prism(vertices=
                [mp.Vector3(mmi_length/2 + taper_length, -gap_between_ports/2-wg_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2, -gap_between_ports/2 - taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2, -gap_between_ports/2 + taper_width/2, -t_Si/2 if THREE_D else 0.0),
                mp.Vector3(mmi_length/2 + taper_length, -gap_between_ports/2+wg_width/2, -t_Si/2 if THREE_D else 0.0)],
                height=t_Si,
                material=mp.Medium(epsilon=n_Si**2))
    ]

    if THREE_D:
        oxide_center = mp.Vector3(z=-0.5 * t_SiO2)
        oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_SiO2)
        oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
        geometry = geometry + oxide_layer

    # Add ports and source

    p1 = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - wg_length/2 - taper_length, gap_between_ports/2),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p2 = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - wg_length/2 - taper_length, -gap_between_ports/2),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p3 = mp.Volume(
        center=mp.Vector3(mmi_length/2 + wg_length/2 + taper_length, gap_between_ports/2),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p4 = mp.Volume(
        center=mp.Vector3(mmi_length/2 + wg_length/2 + taper_length, -gap_between_ports/2),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    src_volume = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - wg_length/2 - taper_length - 0.2, gap_between_ports/2),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )
    
    print(f"p1: {p1.__dict__}")
    print(f"p2: {p2.__dict__}")
    print(f"p3: {p3.__dict__}")
    print(f"p4: {p4.__dict__}")
    print(f"src: {src_volume.__dict__}")

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            size=src_volume.size,
            center=src_volume.center,
            eig_band=1,
            eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell.size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        resolution=resolution)

    # === output flux probes ===
    mode1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p1))
    mode2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p2))
    mode3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p3))
    mode4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p4))

    # === run simulation ===
    sim.run(until_after_sources=100)

    return sim, cell, mode1, mode2, mode3, mode4, src_volume, resolution, pml_layers, geometry


# parameter ranges
mmi_lengths = np.arange(7.5, 8.5, .1)   # µm
mmi_widths  = np.arange(2.15, 2.26, 0.01)   # µm (1.5, 2.8, 0.1)

# allocate output arrays
S31 = np.zeros((len(mmi_widths), len(mmi_lengths)))
S41 = np.zeros((len(mmi_widths), len(mmi_lengths)))

THREE_D = False

# Prepare CSV file with header (only once)
csv_file = "S_parameters_mmi2x2_2D_fine_sweep.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["mmi_width (um)", "mmi_length (um)", "S31", "S41"])

# nested sweeps
for i, mmi_width in enumerate(mmi_widths):
    for j, mmi_length in enumerate(mmi_lengths):
        print(f"Simulation for (width, length)=({mmi_width},{mmi_length})")
        
        # --- run simulation for this (width, length) pair ---
        sim, cell, mode1, mode2, mode3, mode4, src_volume, resolution, pml_layers, geometry = \
            mmi2x2(THREE_D=THREE_D,
                   mmi_length=float(mmi_length),
                   mmi_width=float(mmi_width))   # ← pass width to your function

        # extract eigenmode coefficients
        p1 = sim.get_eigenmode_coefficients(mode1, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]
        p2 = sim.get_eigenmode_coefficients(mode2, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,1]
        p3 = sim.get_eigenmode_coefficients(mode3, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]
        p4 = sim.get_eigenmode_coefficients(mode4, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]

        # store values
        S31[i, j] = abs(p3) / abs(p1)
        S41[i, j] = abs(p4) / abs(p1)

        # append row to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mmi_width, mmi_length, S31[i,j], S41[i,j]])

        print(S31)

        print(S41)