"""Regenerate every table (paper/tables/*.tex) and figure (paper/figures/*.pdf) of the paper.

The numbers come from paper/make_data.py, a single seeded script; figures.py renders them.
"""
import pathlib
import runpy

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper"
runpy.run_path(str(PAPER / "make_data.py"), run_name="__main__")
runpy.run_path(str(PAPER / "figures.py"), run_name="__main__")
print("tables:", sorted(p.name for p in (PAPER / "tables").glob("*.tex")))
