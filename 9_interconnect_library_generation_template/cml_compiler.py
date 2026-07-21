"""
cml_compiler.py

Wraps the CML Compiler CLI (Ansys Interconnect's tool for building custom
photonic component libraries).

More information about the commands can be found in https://optics.ansys.com/hc/en-us/articles/360037138374-Command-line-interface.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path
from .components import PhotonicComponent


class CompilerError(RuntimeError):
    """Raised when a CML Compiler command fails."""


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CmlCompiler:
    """
    Executes CLI commands equivalent to Ansys's CML Compiler tool.

    Responsibilities (per the PDK generation pipeline):
      - create a template library
      - create template components within that library
      - build the final library artifact
    """

    def __init__(
        self,
        library_dir: str = Path(__file__).resolve().parent / "interconnect_library",
        cli_prefix: str = "cml-compiler",
        dry_run: bool = True,
        library_name: str = "foundry"
    ):
        self.library_dir = Path(library_dir)
        self.cli_path = cli_prefix
        self.dry_run = dry_run
        self.history: list[CommandResult] = []
        self.library_name = library_name

    # ------------------------------------------------------------------ #
    # Public pipeline steps
    # ------------------------------------------------------------------ #

    def create_template_library(self) -> CommandResult:
        """Scaffold a new template library on disk."""
        return self._run(
            ["template", "--foundry_name", self.library_name, "-d", str(self.library_dir)],
            simulate=lambda: self.library_dir.mkdir(parents=True, exist_ok=True),
        )
    
    def help(self) -> CommandResult:
        """Display general help menu."""
        return self._run(
            ["help"]
        )
    
    def print_hello_world(self) -> CommandResult:
        """Scaffold a new template library on disk."""
        print( self._run_raw(
            ["echo", "Hello World"]
        ).stdout)


    def validate(self) -> CommandResult:
        """Compile the assembled template library into a final PDK artifact."""
        return self._run(["validate"])

    def build_library(self) -> CommandResult:
        """Compile the assembled template library into a final PDK artifact."""
        return self._run(["library"])

    def run_pipeline(self, library_name: str, components: list["PhotonicComponent"]) -> None:
        """Convenience wrapper: run the full create -> populate -> build flow."""
        self.create_template_library(library_name)
        self.create_template_components(components)
        self.build_library()


    def create_template_components(self, components: list[PhotonicComponent]) -> CommandResult:
        """
        Deploy a batch of template elements from the CML foundry template,
        renaming each to the desired component name.

        Each `component` must expose:
        - component.model : the existing template element name (e.g. "wg_parameterized")
        - component.name   : the desired name for the deployed element (e.g. "wg_straight")

        Produces a single call equivalent to:
            cml-compiler template --element_list model_1, model_2, ... --rename name_1, name_2, ...
        """
        if not components:
            raise ValueError("create_template_components requires at least one component")

        models = [component.model for component in components]
        names = [component.name for component in components]

        args = [
            "template",
            "--element_list",
            ",".join(models),
            "--rename",
            ",".join(names),
        ]
        

        return self._run(args,simulate=lambda: [
            (self.library_dir / component.name).mkdir(parents=True, exist_ok=True)
            for component in components
        ],)

    # ------------------------------------------------------------------ #
    # Internal execution
    # ------------------------------------------------------------------ #

    def _run(
        self,
        args: list[str],
        simulate: Optional[Callable[[], None]] = None,
    ) -> CommandResult:
        command = [self.cli_path, *args]

        if self.dry_run:
            print(f"[dry-run] would execute: {shlex.join(command)}\n")
            if simulate is not None:
                simulate()
            result = CommandResult(command=command, returncode=0, stdout="(mocked)", stderr="")
        else:
            proc = subprocess.run(command, capture_output=True, text=True)
            result = CommandResult(
                command=command,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        self.history.append(result)

        if not result.ok:
            raise CompilerError(
                f"Command failed ({result.returncode}): {shlex.join(command)}\n{result.stderr}"
            )

        return result
    
    def _run_raw(self, args: list[str]) -> CommandResult:
        command = [*args]

        if self.dry_run:
            print(f"[dry-run] would execute: {shlex.join(command)}")
            result = CommandResult(command=command, returncode=0, stdout="(mocked)", stderr="")
        else:
            proc = subprocess.run(command, capture_output=True, text=True)
            result = CommandResult(
                command=command,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        self.history.append(result)

        if not result.ok:
            raise CompilerError(
                f"Command failed ({result.returncode}): {shlex.join(command)}\n{result.stderr}"
            )

        return result
