# Virtual intermediate generation

This package implements the continuity-aware graph augmentation used by
MolSPC. `virtual_intermediates.py` exposes graph interpolation, SMILES
reconstruction, and RDKit calculation of QED, LogP, MW, HBA, HBD, TPSA, and
RB. `generate_pairs.py` is the command-line wrapper for a property CSV.

```bash
python -m molecule_augmentation.generate_pairs \
  --input path/to/property.csv --property tpsa \
  --output data/augmented_pairs.csv --percentage 0.10
```

The input must contain `smiles` and the selected property column. Output rows
contain the source, interpolation partner, generated SMILES, and interpolated
property. For optimization training, use the top-level
`data_finetune_molopt.py --augment` command so generated rows receive all
seven property targets and the standard MolSPC directory layout.
