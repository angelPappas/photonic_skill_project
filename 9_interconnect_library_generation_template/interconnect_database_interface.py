"""Contains class that serves as an intermediate between the database and the Interconnect Library generation class."""

from pathlib import Path
from .components import MMI, DirectionalCoupler, Waveguide, PhotonicComponent

# Key = photonic component type. Value = Interconnect model
COMPONENT_MODEL_MAPPING = {
    "mmi": {"class_name": MMI,
            "interconnect_model": "spar_fixed"},
    "dc": {"class_name": DirectionalCoupler,
            "interconnect_model": "spar_fixed"},
    "wg": {"class_name": Waveguide,
            "interconnect_model": "wg_parameterized"}
}

class InterconnectDatabaseInterface():

    def __init__(self, database_dir=Path(__file__).resolve().parent / "database"):
        self.database_dir = database_dir

    def parse_database(self)->list[PhotonicComponent]:
        components = []

        for directory in self.database_dir.rglob("*"):
            if not directory.is_dir():
                continue

            for component_type, values in COMPONENT_MODEL_MAPPING.items():
                if (component_type + "_") in directory.name:
                    cls = values["class_name"]
                    components.append(
                        cls(
                            name=directory.name,
                            model=values["interconnect_model"],
                            database_path=directory,
                        )
                    )

        return components
