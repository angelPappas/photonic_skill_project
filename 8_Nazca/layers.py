"""
Layer and cross-section (xsection) setup.

Importing this module registers the layers and xsections within the running nazca session.
"""

import nazca as nz

# --- Layers -----------------------------
nz.add_layer(name='WVG', layer=1)
nz.add_layer(name='NDP', layer=2)
nz.add_layer(name='NOP', layer=3)
nz.add_layer(name='PDP', layer=4)
nz.add_layer(name='FIL', layer=5)
nz.add_layer(name='FIS', layer=6)
nz.add_layer(name='MTL', layer=7)
nz.add_layer(name='EDG', layer=100)

# --- Default geometric values -----------
# Set of default geometrical values
wg_width = 2.0
mtl_width = 5.0
heater_width = 1.5
edge_width = 10

# --- Waveguide (WVG) xsection -----------
# Add interconnects or x-sections to nazca design
nz.add_xsection('WVG')
nz.add_layer2xsection(xsection='WVG', layer='WVG')
nz.add_layer2xsection(xsection='WVG', layer='FIL', growx=5, growy=5)
wvg = nz.interconnects.Interconnect(xs='WVG', width=wg_width)

# --- N-doped (NMT) xsection -----------
nz.add_xsection('NMT')
nz.add_layer2xsection(xsection='NMT', layer='NDP')
nz.add_layer2xsection(xsection='NMT', layer='MTL', growx=-3, growy=-3)
nmt = nz.interconnects.Interconnect(xs='NMT', )

# --- P-doped (PMT) xsection -----------
nz.add_xsection('PMT')
nz.add_layer2xsection(xsection='PMT', layer='NDP', growx=1, growy=1)
nz.add_layer2xsection(xsection='PMT', layer='NOP')
nz.add_layer2xsection(xsection='PMT', layer='PDP', growx=-5, growy=-1)
nz.add_layer2xsection(xsection='PMT', layer='MTL', growx=-1.5, growy=-1.5)
pmt = nz.interconnects.Interconnect(xs='PMT', )

# --- Metal (MTL) xsection -----------
nz.add_xsection('MTL')
nz.add_layer2xsection(xsection='MTL', layer='FIL')
nz.add_layer2xsection(xsection='MTL', layer='MTL')
mtl = nz.interconnects.Interconnect(xs='MTL', width=mtl_width)

# --- Heater / phase-shifter (PHL) xsection -----------
nz.add_xsection('PHL')
nz.add_layer2xsection(xsection='PHL', layer='WVG')
nz.add_layer2xsection(xsection='PHL', layer='FIS', growx=2.5, growy=2.5)
nz.add_layer2xsection(xsection='PHL', layer='MTL', growx=-.5, growy=-.5)
phl = nz.interconnects.Interconnect(xs='PHL', width=heater_width)

# --- Waveguide (WVG) xsection -----------
nz.add_xsection('EDG')
nz.add_layer2xsection(xsection='EDG', layer='EDG')
edg = nz.interconnects.Interconnect(xs='EDG', width=edge_width)