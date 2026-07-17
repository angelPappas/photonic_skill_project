"""
Default parameters for a full circuit build (equivalent to the constants
defined at the top of notebook cell 11), plus the bond-pad name mapping
used to wire heaters and photodiodes to the die's edge pads.

Keeping these here (instead of scattered across cells or hidden as
module-level globals) means every component call in ``build.py`` gets its
geometry explicitly, rather than implicitly reading a variable defined
several cells earlier.
"""

# --- Default component geometry ------------------------------------------
coupler_gap = 4
coupler_length = 150
mzm_arm_distance = 150
coupler_radius = 50
heater_length = 200
ring_radius = 80
photodiode_length = 150
photodiode_pcon_width = 10
photodiode_ncon_width = 15
cell_dim = 2500

# --- Bond-pad mapping ------------------------------------------------------
# Maps each MZM heater terminal / photodiode terminal to the physical pad
# name on the product cell's edge (see components/product_cell.py).
bb_pad_list = {
    'H1_in': 'BOT_5',
    'H2_in': 'BOT_4',
    'H3_in': 'BOT_3',
    'H4_in': 'BOT_2',
    'H5_in': 'BOT_1',
    'H6_in': 'TOP_1',
    'H7_in': 'TOP_2',
    'H8_in': 'TOP_3',
    'H9_in': 'TOP_4',
    'H10_in': 'TOP_5',
    'H1_out': 'BOT_9',
    'H2_out': 'BOT_10',
    'H3_out': 'BOT_11',
    'H4_out': 'BOT_12',
    'H5_out': 'BOT_13',
    'H6_out': 'TOP_13',
    'H7_out': 'TOP_12',
    'H8_out': 'TOP_11',
    'H9_out': 'TOP_10',
    'H10_out': 'TOP_9',
    'PD1': 'BOT_16',
    'PD2': 'BOT_17',
    'PD3': 'BOT_18',
    'PD4': 'TOP_22',
    'PD5': 'TOP_21',
    'PD_GND': 'TOP_14',
}
