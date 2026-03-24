"""Shared fixtures for RTMScore tests."""

import os

import numpy as np
import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "example")
TRAINED_MODELS_DIR = os.path.join(REPO_ROOT, "trained_models")


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pocket_pdb_path():
    """Pre-extracted pocket PDB."""
    return os.path.join(EXAMPLE_DIR, "1qkt_p_pocket_10.0.pdb")


@pytest.fixture
def protein_pdb_path():
    """Full protein PDB."""
    return os.path.join(EXAMPLE_DIR, "1qkt_p.pdb")


@pytest.fixture
def decoys_sdf_path():
    """Multi-molecule SDF with 61 decoys."""
    return os.path.join(EXAMPLE_DIR, "1qkt_decoys.sdf")


@pytest.fixture
def ref_ligand_sdf_path():
    """Single reference ligand SDF."""
    return os.path.join(EXAMPLE_DIR, "1qkt_l.sdf")


@pytest.fixture
def model1_path():
    """Path to the first trained model checkpoint."""
    return os.path.join(TRAINED_MODELS_DIR, "rtmscore_model1.pth")


# ---------------------------------------------------------------------------
# Molecule fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_ligand_mol(ref_ligand_sdf_path):
    """RDKit Mol of the reference ligand (1qkt_l)."""
    from RTMScore.feats.mol2graph_rdmda_res import load_mol
    return load_mol(
        ref_ligand_sdf_path,
        explicit_H=False,
        use_chirality=True,
    )


@pytest.fixture
def ethanol_mol():
    """Ethanol molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    return mol


@pytest.fixture
def benzene_mol():
    """Benzene molecule with 3D coordinates and aromaticity."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    return mol


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_ligand_graph(ref_ligand_mol):
    """PyG Data graph of the reference ligand."""
    from RTMScore.feats.mol2graph_rdmda_res import mol_to_graph
    return mol_to_graph(
        ref_ligand_mol,
        explicit_H=False,
        use_chirality=True,
    )


@pytest.fixture
def pocket_graph(pocket_pdb_path):
    """PyG Data graph of the pocket protein (via load_mol, same as VSDataset)."""
    from RTMScore.feats.mol2graph_rdmda_res import load_mol, prot_to_graph
    pocket_mol = load_mol(
        pocket_pdb_path, explicit_H=False, use_chirality=False,
    )
    return prot_to_graph(pocket_mol, 10.0)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

DEFAULT_MODEL_KWARGS = {
    "num_node_featsp": 41,
    "num_node_featsl": 41,
    "num_edge_featsp": 5,
    "num_edge_featsl": 10,
    "hidden_dim0": 128,
    "hidden_dim": 128,
    "n_gaussians": 10,
    "dropout_rate": 0.10,
    "dist_threhold": 5,
}


@pytest.fixture
def model_kwargs():
    """Default model hyperparameters."""
    return DEFAULT_MODEL_KWARGS.copy()


@pytest.fixture
def untrained_model(model_kwargs):
    """An untrained RTMScore model."""
    from RTMScore.model.model2 import (
        GraphTransformer,
        RTMScore,
    )

    ligmodel = GraphTransformer(
        in_channels=model_kwargs["num_node_featsl"],
        edge_features=model_kwargs["num_edge_featsl"],
        num_hidden_channels=model_kwargs["hidden_dim0"],
        activ_fn=th.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply="batch",
        dropout_rate=0.15,
        num_layers=6,
    )
    protmodel = GraphTransformer(
        in_channels=model_kwargs["num_node_featsp"],
        edge_features=model_kwargs["num_edge_featsp"],
        num_hidden_channels=model_kwargs["hidden_dim0"],
        activ_fn=th.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply="batch",
        dropout_rate=0.15,
        num_layers=6,
    )
    return RTMScore(
        ligmodel,
        protmodel,
        in_channels=model_kwargs["hidden_dim0"],
        hidden_dim=model_kwargs["hidden_dim"],
        n_gaussians=model_kwargs["n_gaussians"],
        dropout_rate=model_kwargs["dropout_rate"],
        dist_threhold=model_kwargs["dist_threhold"],
    )


@pytest.fixture
def trained_model(untrained_model, model1_path):
    """RTMScore model loaded with trained weights."""
    checkpoint = th.load(
        model1_path,
        map_location=th.device("cpu"),
    )
    untrained_model.load_state_dict(checkpoint["model_state_dict"])
    return untrained_model


# ---------------------------------------------------------------------------
# Dataset / loader fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vs_dataset(decoys_sdf_path, pocket_pdb_path):
    """VSDataset over the 1qkt decoys."""
    from RTMScore.data.data import VSDataset
    return VSDataset(
        ligs=decoys_sdf_path,
        prot=pocket_pdb_path,
        cutoff=10.0,
        gen_pocket=False,
        explicit_H=False,
        use_chirality=True,
        parallel=False,
    )


@pytest.fixture
def small_dataloader(vs_dataset):
    """DataLoader over the first 5 items of the VS dataset."""
    from torch.utils.data import DataLoader, Subset
    from RTMScore.model.utils import collate

    subset = Subset(vs_dataset, list(range(min(5, len(vs_dataset)))))
    return DataLoader(
        dataset=subset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )


@pytest.fixture
def full_dataloader(vs_dataset):
    """DataLoader over the full VS dataset."""
    from torch.utils.data import DataLoader
    from RTMScore.model.utils import collate

    return DataLoader(
        dataset=vs_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
