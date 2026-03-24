"""Tests for RTMScore.data.data.

Covers VSDataset (SDF loading, mol-list input, PyG Data input)
and PDBbindDataset (graph-list input, train/test split).
"""

import numpy as np
import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Data

from RTMScore.data.data import PDBbindDataset, VSDataset
from RTMScore.feats.mol2graph_rdmda_res import mol_to_graph


# ===================================================================
# VSDataset — from SDF file
# ===================================================================

class TestVSDatasetFromSDF:
    def test_length(self, vs_dataset):
        assert len(vs_dataset) == 61

    def test_ids_first5(self, vs_dataset):
        assert vs_dataset.ids[:5] == [
            "1qkt_119-0",
            "1qkt_132-1",
            "1qkt_144-2",
            "1qkt_150-3",
            "1qkt_31-4",
        ]

    def test_getitem_returns_triple(self, vs_dataset):
        sid, gl, gp = vs_dataset[0]
        assert isinstance(sid, str)
        assert isinstance(gl, Data)
        assert isinstance(gp, Data)

    def test_item_0_ligand_graph_shape(self, vs_dataset):
        _, gl, _ = vs_dataset[0]
        assert gl.num_nodes == 20
        assert gl.num_edges == 46

    def test_item_0_protein_graph_shape(self, vs_dataset):
        _, _, gp = vs_dataset[0]
        assert gp.num_nodes == 79

    def test_protein_graph_shared(self, vs_dataset):
        """All items in VS dataset share the same protein graph."""
        _, _, gp0 = vs_dataset[0]
        _, _, gp1 = vs_dataset[1]
        assert gp0 is gp1

    def test_all_ligand_graphs_valid(self, vs_dataset):
        for i in range(len(vs_dataset)):
            _, gl, _ = vs_dataset[i]
            assert gl.num_nodes > 0
            assert gl.num_edges >= 0
            assert gl.atom is not None
            assert gl.pos is not None
            assert gl.bond is not None

    def test_ligand_feature_dims(self, vs_dataset):
        _, gl, _ = vs_dataset[0]
        assert gl.atom.shape[1] == 41
        assert gl.pos.shape[1] == 3
        assert gl.bond.shape[1] == 10


# ===================================================================
# VSDataset — from mol list
# ===================================================================

class TestVSDatasetFromMolList:
    def test_mol_list_input(self, pocket_pdb_path):
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)

        ds = VSDataset(
            ligs=[mol, mol],
            prot=pocket_pdb_path,
            cutoff=10.0,
            gen_pocket=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
        )
        assert len(ds) == 2

    def test_pyg_graph_list_input(self, pocket_pdb_path, ethanol_mol):
        gl = mol_to_graph(
            ethanol_mol, explicit_H=False, use_chirality=True,
        )
        ds = VSDataset(
            ligs=[gl, gl, gl],
            prot=pocket_pdb_path,
            cutoff=10.0,
            gen_pocket=False,
            explicit_H=False,
            use_chirality=True,
            parallel=False,
        )
        assert len(ds) == 3


# ===================================================================
# VSDataset — SDF splitting
# ===================================================================

class TestVSDatasetSplitting:
    def test_sdf_split_count(self, vs_dataset, decoys_sdf_path):
        """Dataset length should match number of $$$$ blocks."""
        with open(decoys_sdf_path) as f:
            content = f.read()
        n_blocks = content.count("$$$$")
        # Some molecules may fail parsing -> len(ds) <= n_blocks
        assert len(vs_dataset) <= n_blocks
        assert len(vs_dataset) > 0


# ===================================================================
# PDBbindDataset
# ===================================================================

class TestPDBbindDataset:
    @pytest.fixture
    def simple_pdbbind(self):
        """A small PDBbindDataset from arrays."""
        ids = np.array(["pdb1", "pdb2", "pdb3", "pdb4", "pdb5"])
        graphs_l = [
            Data(
                edge_index=th.tensor([[0, 1], [1, 0]], dtype=th.long),
                atom=th.randn(3, 41),
                pos=th.randn(3, 3),
                bond=th.randn(2, 10),
                num_nodes=3,
            )
            for _ in range(5)
        ]
        graphs_p = [
            Data(
                edge_index=th.tensor([[0], [1]], dtype=th.long),
                feats=th.randn(4, 41),
                pos=th.randn(4, 24, 3),
                edge_feats=th.randn(1, 5),
                num_nodes=4,
            )
            for _ in range(5)
        ]
        return PDBbindDataset(ids=ids, ligs=graphs_l, prots=graphs_p)

    def test_length(self, simple_pdbbind):
        assert len(simple_pdbbind) == 5

    def test_getitem(self, simple_pdbbind):
        pid, gl, gp = simple_pdbbind[0]
        assert pid == "pdb1"
        assert isinstance(gl, Data)
        assert isinstance(gp, Data)

    def test_train_test_split_sizes(self, simple_pdbbind):
        train_idx, val_idx = simple_pdbbind.train_and_test_split(
            valnum=2, seed=0,
        )
        assert len(train_idx) == 3
        assert len(val_idx) == 2
        assert len(set(train_idx) & set(val_idx)) == 0

    def test_train_test_split_fraction(self, simple_pdbbind):
        train_idx, val_idx = simple_pdbbind.train_and_test_split(
            valfrac=0.4, seed=0,
        )
        assert len(val_idx) == 2
        assert len(train_idx) == 3

    def test_train_test_split_reproducible(self, simple_pdbbind):
        t1, v1 = simple_pdbbind.train_and_test_split(
            valnum=2, seed=42,
        )
        t2, v2 = simple_pdbbind.train_and_test_split(
            valnum=2, seed=42,
        )
        np.testing.assert_array_equal(v1, v2)

    def test_mismatched_lengths_raises(self):
        ids = np.array(["a", "b"])
        graphs = [
            Data(
                edge_index=th.tensor([[0], [1]], dtype=th.long),
                num_nodes=2,
            )
            for _ in range(3)
        ]
        with pytest.raises(AssertionError):
            PDBbindDataset(ids=ids, ligs=graphs, prots=graphs)
