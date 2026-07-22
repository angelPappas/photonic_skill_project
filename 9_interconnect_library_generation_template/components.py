from abc import ABC, abstractmethod
from pathlib import Path
import csv
import json

def convert_csv_to_json(csv_path: Path, json_path: Path) -> None:
    """
    Convert a CSV file into a JSON array.

    Parameters
    ----------
    csv_path
        Path to the input CSV file.
    json_path
        Path where the JSON file will be written.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        data = {field: [] for field in reader.fieldnames or []}

        for row in reader:
            for field in data:
                data[field].append(row[field])

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

class PhotonicComponent(ABC):
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.model = kwargs["model"]
        self.database_path = kwargs["database_path"]
        self.interconnect_path = kwargs.get("interconnect_path") / self.name

    @abstractmethod
    def apply_changes():
        pass


class MMI(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_changes(self):
        convert_csv_to_json(
            json_path=self.interconnect_path/self.name/f"{self.name}.json",
            csv_path=self.database_path/f"{self.name}.csv")

class DirectionalCoupler(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def apply_changes(self):
        convert_csv_to_json(
            json_path=self.interconnect_path/self.name/f"{self.name}.json",
            csv_path=self.database_path/f"{self.name}.csv")

class Waveguide(PhotonicComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def apply_changes(self):
        convert_csv_to_json(
            json_path=self.interconnect_path/self.name/f"{self.name}.json",
            csv_path=self.database_path/f"{self.name}.csv")
