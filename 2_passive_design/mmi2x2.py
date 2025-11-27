import meep as mp
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
from tqdm import tqdm


def meep_cosine_sbend(
    x0: float=0,
    y0: float=0,
    wg_width: float=0.4,
    t_Si: float=0.22,
    s_bend_length: float=4.0,
    s_bend_width: float=1.0,
    resolution=100,
) -> mp.Prism:
    """
    Build a cosine S-bend waveguide in Meep.
    """

    # ---- 1. Parametric cosine curve ----
    N = resolution
    u = np.linspace(0, 1, N)

    x = x0 + s_bend_length * u
    y = y0 - s_bend_width * np.cos(np.pi * u) / 2 + s_bend_width / 2

    # First we add the lower part of the bend, from left to right, and then the upper part from right to left
    poly_points = [mp.Vector3(x[i], y[i] - wg_width/2,  -t_Si/2) for i in range(N)] + \
                  [mp.Vector3(x[i], y[i] + wg_width/2,  -t_Si/2) for i in reversed(range(N))]

    n_Si = 3.48
    prism = mp.Prism(
        vertices=poly_points,
        height=t_Si,
        material=mp.Medium(epsilon=n_Si**2)
    )

    return prism


# === simulation parameters ===
wavelength = 1.55
fcen = 1/wavelength
df = 0.2*fcen
n_Si = 3.48  # silicon
n_SiO2 = 1.444  # SiO2

def mmi2x2(
        wg_length: float=3, wg_width: float = 0.5, taper_length: float=8, taper_width: float=0.6, mmi_width: float=2.19, mmi_length: float=7.7,
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

    gap_between_ports = 2.19/2 # I am fixing the gap, before was mmi_width/2
    
    # The y-coordinate of the center of the waveguides is at +- gap_between_ports/2
    # If I make the monitors too wide, they will overlap! This is why I choose them to be min(gap_between_ports, 2),
    # making sure they stay on their half of the simulation region (y>0 or y<0).
    monitor_width = wg_width + 0.3

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


def mmi2x2_bend_waveguides(
        straight_wg_length: float=2.0, wg_width: float = 0.4, s_bend_length: float=4.0, s_bend_width: float=1.0, 
        mmi_width: float=1.0, mmi_length: float=3.0, gap: float = 0.2, THREE_D: bool=False):

    THREE_D = THREE_D

    resolution = 30    # pixels/µm

    t_SiO2 = 1.0
    t_Si = 0.22 if THREE_D else 20
    t_air = 0.78

    # geometry
    wg_width = wg_width
    straight_wg_length = straight_wg_length

    s_bend_length = s_bend_length
    s_bend_width = s_bend_width

    mmi_width = mmi_width
    mmi_length = mmi_length

    gap = gap
    
    # The y-coordinate of the center of the waveguides is at +- gap_between_ports/2
    # If I make the monitors too wide, they will overlap! This is why I choose them to be min(gap_between_ports, 2),
    # making sure they stay on their half of the simulation region (y>0 or y<0).
    monitor_width = 2

    dpml = 1
    cell_thickness = (dpml + t_SiO2 + t_Si + t_air + dpml) if THREE_D else 0

    oxide = mp.Medium(epsilon=n_SiO2**2)

    # === simulation cell ===
    sx = straight_wg_length + s_bend_length + mmi_length + s_bend_length + straight_wg_length
    sy = s_bend_width + mmi_width + s_bend_width + 3
    cell = mp.Volume(
        size=mp.Vector3(sx, sy, cell_thickness),
        dims=3
    )

    pml_layers = [mp.PML(1.0)]

    N = 80
    prism1 = meep_cosine_sbend(
            x0=-mmi_length/2 - s_bend_length,
            y0=(gap/2 + wg_width/2 + s_bend_width),
            wg_width=0.4,
            t_Si=t_Si,
            s_bend_length=4.0,
            s_bend_width=-s_bend_width,
            resolution=N)

    prism2 = meep_cosine_sbend(
            x0=-mmi_length/2 - s_bend_length,
            y0=-(gap/2 + wg_width/2 + s_bend_width),
            wg_width=0.4,
            t_Si=t_Si,
            s_bend_length=4.0,
            s_bend_width=s_bend_width,
            resolution=N)
    
    prism3 = meep_cosine_sbend(
            x0=mmi_length/2,
            y0=-(gap/2 + wg_width/2),
            wg_width=0.4,
            t_Si=t_Si,
            s_bend_length=4.0,
            s_bend_width=-s_bend_width,
            resolution=N)
    
    prism4 = meep_cosine_sbend(
            x0=mmi_length/2,
            y0=(gap/2 + wg_width/2),
            wg_width=0.4,
            t_Si=t_Si,
            s_bend_length=4.0,
            s_bend_width=s_bend_width,
            resolution=N)

    # === geometry definition ===
    geometry = [
        # multimode region
        mp.Block(size=mp.Vector3(mmi_length, mmi_width, t_Si),
                center=mp.Vector3(0,0),
                material=mp.Medium(epsilon=n_Si**2)),
        prism1,
        prism2,
        prism3,
        prism4,

        mp.Block(size=mp.Vector3(straight_wg_length, wg_width, t_Si),
                center=mp.Vector3(-mmi_length/2 - s_bend_length - straight_wg_length/2,gap/2 + wg_width/2 + s_bend_width),
                material=mp.Medium(epsilon=n_Si**2)),
        mp.Block(size=mp.Vector3(straight_wg_length, wg_width, t_Si),
                center=mp.Vector3(-mmi_length/2 - s_bend_length - straight_wg_length/2,-(gap/2 + wg_width/2 + s_bend_width)),
                material=mp.Medium(epsilon=n_Si**2)),
        mp.Block(size=mp.Vector3(straight_wg_length, wg_width, t_Si),
                center=mp.Vector3(mmi_length/2 + s_bend_length + straight_wg_length/2,-(gap/2 + wg_width/2 + s_bend_width)),
                material=mp.Medium(epsilon=n_Si**2)),
        mp.Block(size=mp.Vector3(straight_wg_length, wg_width, t_Si),
                center=mp.Vector3(mmi_length/2 + s_bend_length + straight_wg_length/2,gap/2 + wg_width/2 + s_bend_width),
                material=mp.Medium(epsilon=n_Si**2)),
    ]

    if THREE_D:
        oxide_center = mp.Vector3(z=-0.5 * t_SiO2 -0.5*t_Si)
        oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_SiO2)
        oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
        geometry = geometry + oxide_layer

    # Add ports and source

    p1 = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - s_bend_length - straight_wg_length/2,gap/2 + wg_width/2 + s_bend_width),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p2 = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - s_bend_length - straight_wg_length/2,-(gap/2 + wg_width/2 + s_bend_width)),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p3 = mp.Volume(
        center=mp.Vector3(mmi_length/2 + s_bend_length + straight_wg_length/2,-(gap/2 + wg_width/2 + s_bend_width)),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3
        )

    p4 = mp.Volume(
        center=mp.Vector3(mmi_length/2 + s_bend_length + straight_wg_length/2,gap/2 + wg_width/2 + s_bend_width),
        size=mp.Vector3(0, monitor_width, t_Si),
        dims=3,
        )

    src_volume = mp.Volume(
        center=mp.Vector3(-mmi_length/2 - s_bend_length - straight_wg_length/2 - 0.2,gap/2 + wg_width/2 + s_bend_width),
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

    # TODO: look into the monitor's frequency response to see how to get the bandwidth

    # === output flux probes ===
    mode1 = sim.add_mode_monitor(fcen, df, 21, mp.ModeRegion(volume=p1))
    mode2 = sim.add_mode_monitor(fcen, df, 21, mp.ModeRegion(volume=p2))
    mode3 = sim.add_mode_monitor(fcen, df, 21, mp.ModeRegion(volume=p3))
    mode4 = sim.add_mode_monitor(fcen, df, 21, mp.ModeRegion(volume=p4))
    # mode1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p1))
    # mode2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p2))
    # mode3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p3))
    # mode4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p4))

    # === run simulation ===
    sim.run(until_after_sources=100)

    return sim, cell, mode1, mode2, mode3, mode4, src_volume, resolution, pml_layers, geometry


if __name__ == "__main__":

    THREE_D = True

    # # parameter ranges
    # mmi_lengths = np.arange(3.3, 4.0, .05)   # µm

    # S11 = np.zeros(len(mmi_lengths))
    # S21 = np.zeros(len(mmi_lengths))
    # S31 = np.zeros(len(mmi_lengths))
    # S41 = np.zeros(len(mmi_lengths))


    # # Prepare CSV file with header (only once)
    # csv_file = "S_parameters_mmi2x2_with_bends_3D.csv"
    # if not os.path.exists(csv_file):
    #     with open(csv_file, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["mmi_length (um)", "S31", "S41"])


    # for j, mmi_length in enumerate(mmi_lengths):
    #     print(f"Simulation for (width, length)=({mmi_length})")
        
    #     # --- run simulation for this (width, length) pair ---
    #     sim, cell, mode1, mode2, mode3, mode4, src_volume, resolution, pml_layers, geometry = \
    #         mmi2x2_bend_waveguides(THREE_D=THREE_D,
    #             mmi_length=float(mmi_length))

    #     # extract eigenmode coefficients
    #     p1 = sim.get_eigenmode_coefficients(mode1, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]

    #     p1_ref = sim.get_eigenmode_coefficients(mode1, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,1]
        
    #     p2 = sim.get_eigenmode_coefficients(mode2, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,1]
    #     p3 = sim.get_eigenmode_coefficients(mode3, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]
    #     p4 = sim.get_eigenmode_coefficients(mode4, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,0,0]

    #     # store values
    #     S11[j] = abs(p1_ref) / abs(p1)
    #     S21[j] = abs(p2) / abs(p1)
    #     S31[j] = abs(p3) / abs(p1)
    #     S41[j] = abs(p4) / abs(p1)

    #     # append row to CSV
    #     with open(csv_file, mode='a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([mmi_length, S31[j], S41[j]])

    #     print(S31)

    #     print(S41)

    # --- run simulation for this (width, length) pair ---
    sim, cell, mode1, mode2, mode3, mode4, src_volume, resolution, pml_layers, geometry = \
        mmi2x2_bend_waveguides(THREE_D=THREE_D,
            mmi_length=3.15)

    # Prepare CSV file with header (only once)
    csv_file = "S_parameters_mmi2x2_with_bends_3D_3p15.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength (um)", "S11", "S21", "S31", "S41"])

    freqs = mp.get_eigenmode_freqs(mode1)
    wavelengths = 1/np.array(freqs)

    # extract eigenmode coefficients
    p1 = sim.get_eigenmode_coefficients(mode1, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,:,0]
    p1_refl = sim.get_eigenmode_coefficients(mode1, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,:,1]
    p2 = sim.get_eigenmode_coefficients(mode2, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,:,1]
    p3 = sim.get_eigenmode_coefficients(mode3, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,:,0]
    p4 = sim.get_eigenmode_coefficients(mode4, [1], eig_parity=mp.NO_PARITY if THREE_D else mp.EVEN_Y + mp.ODD_Z).alpha[0,:,0]

    # store values
    S11 = abs(p1_refl) / abs(p1)
    S21 = abs(p2) / abs(p1)
    S31 = abs(p3) / abs(p1)
    S41 = abs(p4) / abs(p1)

    # append row to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([wavelengths, S11, S21, S31, S41])
    