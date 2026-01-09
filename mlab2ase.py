#!/usr/bin/env python3
"""
mlab2ase.py — Parse an MLab-style multi-configuration text file into an ASE trajectory (EXTXYZ/XYZ).

Per configuration, the parser extracts:
- Primitive lattice vectors (Å) -> Atoms.cell
- Atom types and atom numbers   -> Atoms.symbols
- Atomic positions (Å)          -> Atoms.positions
- Total energy (eV)             -> atoms.info["energy"]
- Forces (eV/Å)                 -> atoms.arrays["force"]   (written once, like the original script)
- Stress (kbar) (optional)      -> atoms.info["stress"] as a 3x3 tensor in eV/Å^3 (physically correct)

CLI:
  python mlab2ase.py ML_AB -o mlab_dataset.xyz
  python mlab2ase.py ML_AB -o mlab_dataset.extxyz --format extxyz
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms, io


logger = logging.getLogger("mlab2ase")

# Physically correct conversion:
# 1 kbar = 1e8 Pa
# 1 eV/Å^3 = 1.602176634e11 Pa
# => 1 kbar = 1e8 / 1.602176634e11 = 6.241509074e-4 eV/Å^3
KBAR_TO_EV_PER_A3 = 6.241509074e-4

_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"


@dataclass(frozen=True)
class ParseOptions:
    lattice_zero_threshold: float = 0.005  # set tiny lattice elements to 0.0
    pbc: bool = True


# ----------------------------- parsing helpers -----------------------------

def _find_section_block(block_text: str, header_regex: str) -> Optional[str]:
    """
    Extract text between a section header and the next ===== or ---- delimiter.
    Returns inner text or None.
    """
    pattern = re.compile(
        rf"{header_regex}.*?(?:-+|=+)\s*\n(.*?)\n\s*=+",
        flags=re.S,
    )
    m = pattern.search(block_text)
    return None if m is None else m.group(1)


def _parse_matrix_3x3(text: str) -> np.ndarray:
    vals = np.fromstring(text, sep=" ")
    if vals.size != 9:
        raise ValueError(f"Expected 9 numbers for 3x3 lattice, got {vals.size}")
    return vals.reshape(3, 3)


def _parse_nx3(text: str, n: int, what: str) -> np.ndarray:
    vals = np.fromstring(text, sep=" ")
    if vals.size != 3 * n:
        raise ValueError(f"Expected {3*n} numbers for {what}, got {vals.size}")
    return vals.reshape(n, 3)


def _parse_symbols(type_block_text: str) -> List[str]:
    symbols: List[str] = []
    for raw in type_block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sym, count = parts[0], parts[1]
        try:
            n = int(count)
        except ValueError as exc:
            raise ValueError(f"Invalid atom count '{count}' on line: {raw!r}") from exc
        symbols.extend([sym] * n)
    if not symbols:
        raise ValueError("No atom symbols parsed")
    return symbols


def _parse_energy(block_text: str) -> float:
    """
    Robust energy parsing:
    find 'Total energy (eV)' then take the first float that appears after it in a short window.
    """
    m = re.search(r"Total energy\s*\(eV\)", block_text)
    if m is None:
        raise ValueError("Energy header not found")
    window = block_text[m.start(): m.start() + 600]
    nums = re.findall(_FLOAT_RE, window)
    if not nums:
        raise ValueError("Energy value not found after header")
    return float(nums[0])


def _parse_stress_tensor_ev_a3(block_text: str) -> Optional[np.ndarray]:
    """
    Optional stress parser.

    Reads a 'Stress (kbar)' block and extracts two numeric triples:
      triple 1 -> (xx, yy, zz)
      triple 2 -> (xy, yz, zx)

    Returns a 3x3 symmetric tensor in eV/Å^3:
      [[xx, xy, zx],
       [xy, yy, yz],
       [zx, yz, zz]]
    or None if missing/unparseable.
    """
    s_head = re.search(r"Stress\s*\(kbar\)", block_text)
    if s_head is None:
        return None

    sub = block_text[s_head.start():]
    sub = re.split(r"\*{5,}", sub)[0]  # stop at ***** footer if present

    triples: List[List[float]] = []
    for line in sub.splitlines():
        nums = re.findall(_FLOAT_RE, line)
        if len(nums) == 3:
            triples.append([float(x) for x in nums])

    if len(triples) < 2:
        return None

    xx, yy, zz = triples[0]
    xy, yz, zx = triples[1]

    # Convert kbar -> eV/Å^3
    xx *= KBAR_TO_EV_PER_A3
    yy *= KBAR_TO_EV_PER_A3
    zz *= KBAR_TO_EV_PER_A3
    xy *= KBAR_TO_EV_PER_A3
    yz *= KBAR_TO_EV_PER_A3
    zx *= KBAR_TO_EV_PER_A3

    stress = np.array(
        [
            [xx, xy, zx],
            [xy, yy, yz],
            [zx, yz, zz],
        ],
        dtype=float,
    )
    return stress


# ----------------------------- public API -----------------------------

def read_mlab(path: str | Path, options: ParseOptions = ParseOptions()) -> List[Atoms]:
    """
    Parse MLab-style multi-configuration text into a list of ASE Atoms.

    Incomplete configurations (missing any required section) are skipped with a warning.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    blocks = re.split(r"Configuration num\.", text)
    if len(blocks) <= 1:
        logger.warning("No 'Configuration num.' blocks found in %s", path)

    atoms_list: List[Atoms] = []

    for blk_id, blk in enumerate(blocks[1:], start=1):
        try:
            # lattice
            latt_txt = _find_section_block(blk, r"Primitive lattice vectors\s*\(ang\.\)")
            if latt_txt is None:
                raise ValueError("lattice section not found")
            cell = _parse_matrix_3x3(latt_txt)
            if options.lattice_zero_threshold > 0:
                cell[np.abs(cell) < options.lattice_zero_threshold] = 0.0

            # types -> symbols
            type_txt = _find_section_block(blk, r"Atom types and atom numbers")
            if type_txt is None:
                raise ValueError("atom types section not found")
            symbols = _parse_symbols(type_txt)
            n_atoms = len(symbols)

            # positions
            pos_txt = _find_section_block(blk, r"Atomic positions\s*\(ang\.\)")
            if pos_txt is None:
                raise ValueError("positions section not found")
            positions = _parse_nx3(pos_txt, n_atoms, what="positions")

            # energy
            energy = _parse_energy(blk)

            # forces
            f_txt = _find_section_block(blk, r"Forces\s*\(eV ang\.\^-1\)")
            if f_txt is None:
                raise ValueError("forces section not found")
            forces = _parse_nx3(f_txt, n_atoms, what="forces")

            # stress (optional)
            stress = _parse_stress_tensor_ev_a3(blk)

            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=options.pbc)
            atoms.info["energy"] = float(energy)

            # IMPORTANT: write forces ONCE like your original script
            # This produces Properties=...:force:R:3 (no forces:R:3)
            atoms.arrays["force"] = np.asarray(forces, dtype=float)

            if stress is not None:
                atoms.info["stress"] = stress

            atoms_list.append(atoms)

        except Exception as exc:
            logger.warning("Skipping configuration %d: %s", blk_id, exc)

    return atoms_list


def write_trajectory(out_path: str | Path, atoms_list: Sequence[Atoms], fmt: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(str(out_path), list(atoms_list), format=fmt)


def _print_one_time_dataset_summary(atoms_list: Sequence[Atoms]) -> None:
    """
    Prints the confirmation block once, as requested.
    We treat "ok" as "present for all parsed structures" for required fields,
    and "stress" is ok only if present for all structures (since it's optional in parsing).
    """
    n = len(atoms_list)
    print(f"Number of structures in the training set: {n}")

    # Required fields: lattice, types, positions, energy, forces
    lattice_ok = all(a.cell is not None and np.asarray(a.cell).shape == (3, 3) for a in atoms_list) if n else False
    types_ok = all(len(a) > 0 and all(sym is not None for sym in a.get_chemical_symbols()) for a in atoms_list) if n else False
    positions_ok = all(np.asarray(a.positions).shape == (len(a), 3) for a in atoms_list) if n else False
    energy_ok = all("energy" in a.info for a in atoms_list) if n else False
    forces_ok = all("force" in a.arrays and np.asarray(a.arrays["force"]).shape == (len(a), 3) for a in atoms_list) if n else False

    # Stress is optional; call it ok only if every structure has it and it is 3x3
    stress_ok = all(("stress" in a.info) and (np.asarray(a.info["stress"]).shape == (3, 3)) for a in atoms_list) if n else False

    print(f"  lattice: {'ok' if lattice_ok else 'missing'}")
    print(f"  types: {'ok' if types_ok else 'missing'}")
    print(f"  positions: {'ok' if positions_ok else 'missing'}")
    print(f"  energy: {'ok' if energy_ok else 'missing'}")
    print(f"  forces: {'ok' if forces_ok else 'missing'}")
    print(f"  stress: {'ok' if stress_ok else 'missing'}")


# ----------------------------- CLI -----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert MLab-style text output to ASE trajectory.")
    p.add_argument("input", help="Input text file (e.g. ML_AB).")
    p.add_argument("-o", "--output", default="mlab_dataset.extxyz", help="Output trajectory filename.")
    p.add_argument("--format", default=None, help="ASE format override (e.g. extxyz, xyz).")
    p.add_argument(
        "--lattice-zero-threshold",
        type=float,
        default=0.005,
        help="Set lattice components with abs(value) < threshold to 0.0 (default: 0.005).",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv).")
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    _configure_logging(args.verbose)

    options = ParseOptions(lattice_zero_threshold=float(args.lattice_zero_threshold))

    atoms_list = read_mlab(args.input, options=options)
    if not atoms_list:
        logger.error("No configurations parsed — nothing to write.")
        return 2

    write_trajectory(args.output, atoms_list, fmt=args.format)

    # One-time confirmation printout (requested)
    _print_one_time_dataset_summary(atoms_list)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
