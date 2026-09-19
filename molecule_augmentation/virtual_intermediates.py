"""Generate chemically valid virtual intermediates by graph interpolation.

The implementation follows the method used for the continuity-aware data
augmentation experiments: two nearby molecular graphs are aligned, their
Laplacian eigenvalue representations and node features are interpolated, and the result is
converted back to a sanitized SMILES string.  The module deliberately keeps
the public API small so that it can be used both by the property-data
augmentation script and by molecular optimization fine-tuning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from ogb.utils.features import (
    atom_feature_vector_to_dict,
    atom_to_feature_vector,
    bond_feature_vector_to_dict,
    bond_to_feature_vector,
)
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from scipy.linalg import eigh, qr, svd
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data


PROPERTY_NAMES = ("QED", "LogP", "MW", "HBA", "HBD", "TPSA", "RB")


def calculate_molecular_properties(smiles: str) -> Optional[dict[str, float]]:
    """Return the seven properties used by MolSPC for one SMILES string."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "LogP": float(Descriptors.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "MW": float(Descriptors.MolWt(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "RB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


def smiles_to_graph(smiles: str, y: float = 0.0) -> Data:
    """Encode a SMILES string with the same OGB atom and bond features."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    node_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_to_feature_vector(bond)
        edges.extend(((i, j), (j, i)))
        edge_features.extend((feature, feature))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float),
        smiles=Chem.MolToSmiles(mol, canonical=True),
    )


def _as_graph_dict(graph: Data) -> dict:
    return {
        "edge_index": graph.edge_index.detach().cpu().numpy(),
        "edge_feat": graph.edge_attr.detach().cpu().numpy(),
        "node_feat": graph.x.detach().cpu().numpy(),
        "y": float(graph.y.view(-1)[0].item()),
        "smiles": getattr(graph, "smiles", None),
    }


def _edge_adjacencies(graph: dict) -> list[np.ndarray]:
    """Build one weighted adjacency matrix per bond feature."""
    edge_index = np.asarray(graph["edge_index"])
    edge_feat = np.asarray(graph["edge_feat"], dtype=float) + 1.0
    n = int(graph["node_feat"].shape[0])
    if edge_feat.ndim != 2:
        edge_feat = edge_feat.reshape(-1, 1)
    adjacencies = []
    for feature_idx in range(edge_feat.shape[1]):
        matrix = np.zeros((n, n), dtype=float)
        for edge_idx in range(edge_index.shape[1]):
            i, j = map(int, edge_index[:, edge_idx])
            value = edge_feat[edge_idx, feature_idx]
            matrix[i, j] = matrix[j, i] = value
        adjacencies.append(matrix)
    return adjacencies


def _pad_graphs(a: np.ndarray, b: np.ndarray, xa: np.ndarray, xb: np.ndarray):
    n = max(a.shape[0], b.shape[0])
    a_pad = np.zeros((n, n), dtype=float)
    b_pad = np.zeros((n, n), dtype=float)
    xa_pad = np.zeros((n, xa.shape[1]), dtype=float)
    xb_pad = np.zeros((n, xb.shape[1]), dtype=float)
    a_pad[: a.shape[0], : a.shape[0]] = a
    b_pad[: b.shape[0], : b.shape[0]] = b
    xa_pad[: xa.shape[0]] = xa
    xb_pad[: xb.shape[0]] = xb
    return a_pad, b_pad, xa_pad, xb_pad


def _interpolate_adjacency(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Interpolate two Laplacians after eigenvector alignment."""
    la = np.diag(a.sum(axis=1)) - a
    lb = np.diag(b.sum(axis=1)) - b
    va, ua = eigh(la)
    vb, ub = eigh(lb)
    k = min(len(va), len(vb))
    ua, ub = ua[:, :k], ub[:, :k]
    va, vb = va[:k], vb[:k]
    signs = np.sign(np.sum(ua * ub, axis=0))
    ub = ub * np.where(signs == 0, 1.0, signs)
    left, _, right = svd(ua.T @ ub, full_matrices=False)
    ub = ub @ (left @ right)
    basis, _ = qr((1.0 - gamma) * ua + gamma * ub, mode="economic")
    values = (1.0 - gamma) * va + gamma * vb
    interpolated_l = basis @ np.diag(values) @ basis.T
    interpolated_l = (interpolated_l + interpolated_l.T) / 2.0
    adjacency = np.maximum(0.0, -interpolated_l)
    np.fill_diagonal(adjacency, 0.0)
    # The OGB bond representation is categorical.  SPECTRA quantizes the
    # interpolated Laplacian before reconstructing edges; without this step,
    # numerical noise creates an almost complete graph.
    return np.rint(adjacency).astype(np.int64)


def _reconstruct_graph(graph_a: dict, graph_b: dict, gamma: float, alpha: float) -> dict:
    xa, xb = graph_a["node_feat"], graph_b["node_feat"]
    adj_a, adj_b = _edge_adjacencies(graph_a), _edge_adjacencies(graph_b)
    n_features = min(len(adj_a), len(adj_b))
    if n_features == 0:
        raise ValueError("Molecule graphs have no edge features")

    padded = [_pad_graphs(adj_a[i], adj_b[i], xa, xb) for i in range(n_features)]
    # Recover a node permutation from the first bond-feature channel and share
    # it across all bond feature channels.
    _, _, xa_pad, xb_pad = padded[0]
    cost = ((xa_pad[:, None, :] - xb_pad[None, :, :]) ** 2).mean(axis=2)
    rows, cols = linear_sum_assignment(cost)
    permutation = np.arange(xa_pad.shape[0])
    permutation[rows] = cols
    aligned_xb = xb_pad[permutation]

    interpolated = [_interpolate_adjacency(padded[i][0], padded[i][1][permutation][:, permutation], gamma)
                    for i in range(n_features)]
    n = interpolated[0].shape[0]
    edges, attrs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            values = np.asarray([matrix[i, j] for matrix in interpolated], dtype=np.int64)
            if np.max(values) <= 0:
                continue
            edges.extend(((i, j), (j, i)))
            # OGB bond features are categorical indices.  Rounding after
            # interpolation preserves the representation expected by the
            # model and graph-to-SMILES conversion.
            value = np.maximum(0, values - 1)
            attrs.extend((value, value))

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_feat = np.asarray(attrs, dtype=np.int64)
        # Remove padded nodes that received no interpolated bond. This avoids
        # turning alignment-only padding into disconnected carbon atoms.
        used = np.unique(edge_index)
        remap = {int(old): new for new, old in enumerate(used)}
        edge_index = np.vectorize(remap.__getitem__)(edge_index)
        node_feat = np.rint((1.0 - gamma) * xa_pad + gamma * aligned_xb).astype(np.int64)[used]
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, n_features), dtype=np.int64)
        node_feat = np.rint((1.0 - gamma) * xa_pad + gamma * aligned_xb).astype(np.int64)
    return {
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "node_feat": node_feat,
        "y": (1.0 - gamma) * graph_a["y"] + gamma * graph_b["y"],
    }


def graph_to_smiles(graph: dict) -> str:
    """Convert an interpolated graph back to canonical, sanitized SMILES."""
    mol = Chem.RWMol()
    indices = {}
    for row in np.asarray(graph["node_feat"], dtype=int):
        atom_dict = atom_feature_vector_to_dict(row.tolist())
        atomic_num = atom_dict.get("atomic_num", 6)
        if atomic_num == "misc":
            atomic_num = 6
        atom = Chem.Atom(int(atomic_num))
        if atom_dict.get("formal_charge") != "misc":
            atom.SetFormalCharge(int(atom_dict["formal_charge"]))
        aromatic = atom_dict.get("is_aromatic", False)
        if aromatic != "misc":
            atom.SetIsAromatic(bool(aromatic))
        hybridization = atom_dict.get("hybridization")
        if hybridization and hybridization != "misc":
            try:
                atom.SetHybridization(getattr(Chem.rdchem.HybridizationType, hybridization))
            except (AttributeError, TypeError):
                pass
        indices[len(indices)] = mol.AddAtom(atom)

    added = set()
    edge_index = np.asarray(graph["edge_index"])
    edge_feat = np.asarray(graph["edge_feat"], dtype=int)
    for edge_idx in range(edge_index.shape[1]):
        i, j = map(int, edge_index[:, edge_idx])
        if (i, j) in added or (j, i) in added:
            continue
        bond_dict = bond_feature_vector_to_dict(edge_feat[edge_idx].tolist())
        kind = {
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE,
            "AROMATIC": Chem.BondType.AROMATIC,
        }.get(bond_dict.get("bond_type"), Chem.BondType.SINGLE)
        try:
            mol.AddBond(indices[i], indices[j], kind)
        except RuntimeError:
            continue
        added.update(((i, j), (j, i)))

    result = mol.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, canonical=True)


def interpolate_smiles(
    smiles_a: str,
    smiles_b: str,
    y_a: float = 0.0,
    y_b: float = 1.0,
    gamma: float = 0.1,
    alpha: float = 0.5,
) -> tuple[str, float]:
    """Return one virtual intermediate and its interpolated scalar target."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1")
    graph_a = _as_graph_dict(smiles_to_graph(smiles_a, y_a))
    graph_b = _as_graph_dict(smiles_to_graph(smiles_b, y_b))
    graph = _reconstruct_graph(graph_a, graph_b, gamma=gamma, alpha=alpha)
    return graph_to_smiles(graph), float(graph["y"])


def _property_distance(y: np.ndarray) -> np.ndarray:
    scale = np.std(y) or 1.0
    return np.abs(y[:, None] - y[None, :]) / scale


def augment_molecular_dataset(
    dataframe: pd.DataFrame,
    property_name: str,
    percentage: float = 0.10,
    gamma: float = 0.1,
    alpha: float = 0.5,
    seed: int = 42,
    return_skipped: bool = False,
):
    """Create density-aware virtual intermediates from ``smiles``/property rows.

    The returned table contains the original pair, the interpolation partner,
    and the generated molecule.  Invalid graph reconstructions are skipped.
    Set ``return_skipped`` to also return exception types for diagnostics.
    """
    if "smiles" not in dataframe or property_name not in dataframe:
        raise ValueError("Input data must contain 'smiles' and the selected property")
    if not 0.0 <= percentage:
        raise ValueError("percentage must be non-negative")
    source = dataframe[["smiles", property_name]].dropna().copy()
    source["smiles"] = source["smiles"].astype(str)
    source[property_name] = pd.to_numeric(source[property_name], errors="coerce")
    source = source.dropna().drop_duplicates("smiles").reset_index(drop=True)
    if len(source) < 2 or percentage == 0:
        empty = pd.DataFrame(columns=[
            "original_smiles", "interpolated_with_smiles", "augmented_smiles",
            "original_y", "interpolated_with_y", "augmented_y",
        ])
        return (empty, []) if return_skipped else empty

    rng = np.random.default_rng(seed)
    y = source[property_name].to_numpy(dtype=float)
    distances = _property_distance(y)
    np.fill_diagonal(distances, np.inf)
    # Rare property regions receive more interpolation attempts.
    density = np.exp(-distances).mean(axis=1)
    weights = 1.0 / np.maximum(density, 1e-8)
    weights /= weights.sum()
    count = int(round(len(source) * percentage))
    selected = rng.choice(len(source), size=count, replace=True, p=weights)
    records = []
    skipped = []
    for idx in selected:
        # A close property neighbor can still produce an invalid graph. Try
        # the next closest neighbors, as SPECTRA does, before giving up.
        candidate_limit = min(len(source) - 1, 30)
        candidates = np.argsort(distances[idx])[:candidate_limit]
        last_error = None
        for partner in candidates:
            try:
                augmented, augmented_y = interpolate_smiles(
                    source.at[idx, "smiles"], source.at[partner, "smiles"],
                    y_a=y[idx], y_b=y[partner], gamma=gamma, alpha=alpha,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            skipped.append(type(last_error).__name__ if last_error else "NoCandidate")
            continue
        records.append({
            "original_smiles": source.at[idx, "smiles"],
            "interpolated_with_smiles": source.at[partner, "smiles"],
            "augmented_smiles": augmented,
            "original_y": y[idx],
            "interpolated_with_y": y[partner],
            "augmented_y": augmented_y,
        })
    pairs = pd.DataFrame(records)
    return (pairs, skipped) if return_skipped else pairs
