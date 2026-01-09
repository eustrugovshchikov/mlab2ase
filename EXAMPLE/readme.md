## Bulk α-SiO₂: ML_AB to ASE EXTXYZ Conversion Example

### Short Description

Example demonstrating the conversion of a VASP `ML_AB` training set for bulk a-SiO2 into an
ASE-compatible trajectory. The dataset contains 82 configurations with energies, forces, and
stresses, suitable for training modern GNN interatomic potentials.

---

### Example Command and Output

Run the converter from this directory:

```bash
python mlab2ase.py ML_AB
```

Expected terminal output:

```bash
Number of structures in the training set: 82
  lattice: ok
  types: ok
  positions: ok
  energy: ok
  forces: ok
  stress: ok
```
