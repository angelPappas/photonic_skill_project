# Photonic Skill Project

Photonic integrated circuit (PIC) design and simulation project using [Tidy3D](https://www.flexcompute.com/tidy3d/) and [FEMwell](https://github.com/flaport/femwell) for electromagnetic wave simulations.

## Project Structure

```
photonic_skill_project/
├── 1_pn_phase_shifter_FEMWELL/    # PN phase shifter analysis using FEMwell
├── 2_SiN_single_mode_wg_FEMWELL/  # Silicon nitride single-mode waveguide
├── 3_2x2_splitter_MEEP/           # Passive component designs (MMI, couplers)
├── 4_simplest_transceiver_layout_GDSFactory/            # Layout files (GDS)
├── 5_bend_waveguide_Tidy3D/       # Bent waveguide characterization
├── environment.yml                # Conda environment specification
├── pyproject.toml                 # Python project configuration
└── README.md                      # This file
```

## Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- Python 3.13+

### Installation

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate pmp
```

Or using uv:

```bash
uv sync
```

## Components

### 1. PN Phase Shifter (FEMWELL)

Analysis of reverse-biased PN junction phase shifters using FEMwell mode solver. Calculates:
- Refractive index change due to carrier depletion
- Lπ: π phase shift length for given applied voltages
- Insertion loss

### 2. Silicon Nitride Single-Mode Waveguide

FEMWELL-based single-mode waveguide characterization for SiN platform.
For fixed waveguide thickness, and variable width, calculates effective index of up to 8 modes, and sets the maximum width for which it is single-mode.

### 3. Passive Design

- **MMI 2x2**: Multimode interference splitter design and optimization
- **Directional Coupler**: Coupler design with S-parameter extraction
- **GDS Files**: Layout exports for fabrication

### 4. Bent Waveguide Characterization

Bend loss and mode analysis for curved waveguide structures.

### 5. Transceiver Layout

Implementation of the simplest transceiver layout in GDSFactory.

## Simulation Tools

| Tool | Purpose | Availability
|------|---------|--------------|
| [Tidy3D](https://www.flexcompute.com/tidy3d/) | FDTD simulations (3D) | Subscription required |
| [FEMwell](https://github.com/flaport/femwell) | FEM mode solver (2D cross-section) | Open-source |
| [Meep](https://github.com/LumericalFDTD/meep) | FDTD with MPI parallelization | Open-source |

## Notebooks

- `draft.ipynb` — Initial development and testing
- `tidy3d_example.ipynb` — Tidy3D tutorial examples

## License

MIT