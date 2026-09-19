"""Continuity-aware virtual intermediate generation for MolSPC."""

from .virtual_intermediates import (
    PROPERTY_NAMES,
    calculate_molecular_properties,
    augment_molecular_dataset,
    interpolate_smiles,
)

__all__ = [
    "PROPERTY_NAMES",
    "calculate_molecular_properties",
    "augment_molecular_dataset",
    "interpolate_smiles",
]
