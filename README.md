# mlab2ase
Converter for VASP native ML_AB training set files generated in ML_FF workflows. Translates energies, forces, stresses, and structures directly into ASE-compatible XYZ/EXTXYZ trajectories for modern GNN interatomic potentials such as MACE and NequIP.

---

## Overview

`mlab2ase` is a lightweight, robust **converter for VASP native `ML_AB` training set files**
generated during **machine-learning force field (ML_FF) on-the-fly training**.

It converts `ML_AB` files **directly into ASE-compatible `xyz` / `extxyz` trajectories**, making
them immediately usable for **modern graph neural network (GNN) interatomic potentials**, such as:

- MACE  
- NequIP  
- other ASE-based ML pipelines  

The goal of this script is **clarity, correctness, and minimal assumptions**, rather than a large
framework or opaque abstraction layer.

---

## Theory

### What This Script Is For

During VASP ML_FF on-the-fly training, training data are stored in a native text format (`ML_AB`)
containing multiple atomic configurations with energies, forces, and stresses.

While suitable for VASP, this format is **not directly usable** by most modern ML potential
frameworks, which expect **ASE trajectories**.

This script performs a **direct, transparent translation**:

- no interpolation,
- no reordering,
- no hidden transformations.

### What the Script Does

For each configuration block (identified by `Configuration num.`), the script:

- **Reads**
  - primitive lattice vectors (Å),
  - atom types and atom numbers,
  - atomic positions (Å),
  - total energy (eV),
  - atomic forces (eV/Å),
  - stress tensor (kbar, optional).

- **Converts**
  - stress from `kbar` to **eV/Å³** using a physically correct conversion factor.

- **Constructs**
  - an `ase.Atoms` object with periodic boundary conditions.

- **Writes**
  - all configurations into a single ASE trajectory (`xyz` or `extxyz`).

Incomplete or malformed configurations are skipped safely.

---

## Input Format

The input file must be a **VASP-generated `ML_AB` file**, containing repeated blocks with headers
similar to:

- `Primitive lattice vectors (ang.)`
- `Atom types and atom numbers`
- `Atomic positions (ang.)`
- `Total energy (eV)`
- `Forces (eV ang.^-1)`
- `Stress (kbar)` (optional)

Each configuration must be preceded by:

`Configuration num.`

## Installation

### Requirements

- `Python ≥ 3.8`
- `numpy`  
- `ase`

## Usage
### Basic Usage

`python mlab2ase.py ML_AB -o mlab_dataset.extxyz`

###Extended Usage

```text
python mlab2ase.py ML_AB \
  --output mlab_dataset.xyz \
  --format extxyz \
  --lattice-zero-threshold 0.005 \
  -v
```

## Output Validation

After writing the output trajectory, the script prints a one-time validation summary:

```text
Number of structures in the training set: N
  lattice: ok
  types: ok
  positions: ok
  energy: ok
  forces: ok
  stress: ok
```

##How to Cite

If you use this script in your work, please cite it as:

```bash
@article{Strugovshchikov2025,
  title = {Interfacial behavior from the atomic blueprint: Machine learning-guided design of spatially functionalized $\alpha$-$SiO_2$ surfaces},
  volume = {702},
  ISSN = {0021-9797},
  url = {http://dx.doi.org/10.1016/j.jcis.2025.138943},
  DOI = {10.1016/j.jcis.2025.138943},
  journal = {Journal of Colloid and Interface Science},
  publisher = {Elsevier BV},
  author = {\textbf{Strugovshchikov,  Evgenii} and Mandrolko,  Viktor and Lesnicki,  Dominika and Pastore,  Mariachiara and Chaput,  Laurent and Isaiev,  Mykola},
  year = {2026},
  month = jan,
  pages = {138943}
}
```

The script was developed during the work leading to this publication and reflect the data generation and analysis workflow used therein.
