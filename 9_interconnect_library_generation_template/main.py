from .cml_compiler import CmlCompiler
from .interconnect_database_interface import InterconnectDatabaseInterface
from pathlib import Path
from pprint import pprint

INTERCONNECT_LIBRARY_NAME = "photonics_foundry"

compiler = CmlCompiler(library_name=INTERCONNECT_LIBRARY_NAME)

# ----- 2. Parse Database ---------#
interface = InterconnectDatabaseInterface()

photonic_components = interface.parse_database()

for component in photonic_components:
    pprint([component.name, component.model, component.database_path], indent=2)

compiler.create_template_library()



compiler.create_template_components(photonic_components)