from .cml_compiler import CmlCompiler
from .interconnect_database_interface import InterconnectDatabaseInterface
from pathlib import Path
from pprint import pprint

INTERCONNECT_LIBRARY_NAME = "photonics_foundry"

compiler = CmlCompiler(library_name=INTERCONNECT_LIBRARY_NAME)

# ----- 2. Parse Database & list components ---------#
interface = InterconnectDatabaseInterface()

photonic_components = interface.parse_database()

for component in photonic_components:
    pprint([component.name, component.model], indent=2)

# ----- 3. Launch template ---------#
compiler.create_template_library()


# ----- 4. Create equivalent models ---------#
compiler.create_template_components(photonic_components)

# ----- 5. Apply component-specific changes ---------#
for component in photonic_components:
    component.apply_changes()

# ----- 6. Build library ---------#
compiler.build_library()
