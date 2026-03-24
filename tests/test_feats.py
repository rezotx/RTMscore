"""Tests for RTMScore.feats.mol2graph_rdmda_res.

Covers encoding helpers, atom/bond featurization, and graph
construction for both ligands (mol_to_graph) and proteins
(prot_to_graph).  Numeric values are pinned against the reference
ligand 1qkt_l.sdf and pocket 1qkt_p_pocket_10.0.pdb so that any
dependency upgrade (DGL → PyG, torch version bump, etc.) that silently
changes the graph topology or feature tensors will be caught.
"""

import numpy as np
import pytest
import torch as th
from rdkit import Chem

from RTMScore.feats.mol2graph_rdmda_res import (
    calc_atom_features,
    calc_bond_features,
    load_mol,
    mol_to_graph,
    one_of_k_encoding,
    one_of_k_encoding_unk,
    prot_to_graph,
)


# ===================================================================
# one_of_k_encoding / one_of_k_encoding_unk
# ===================================================================

class TestOneOfKEncoding:
    def test_exact_match(self):
        assert one_of_k_encoding("C", ["C", "N", "O"]) == [
            True, False, False,
        ]

    def test_last_element(self):
        assert one_of_k_encoding("O", ["C", "N", "O"]) == [
            False, False, True,
        ]

    def test_raises_on_unknown(self):
        with pytest.raises(Exception, match="not in allowable set"):
            one_of_k_encoding("X", ["C", "N", "O"])


class TestOneOfKEncodingUnk:
    def test_known_element(self):
        assert one_of_k_encoding_unk("C", ["C", "N", "other"]) == [
            True, False, False,
        ]

    def test_unknown_maps_to_last(self):
        assert one_of_k_encoding_unk("X", ["C", "N", "other"]) == [
            False, False, True,
        ]

    def test_single_element_set(self):
        assert one_of_k_encoding_unk("Z", ["other"]) == [True]


# ===================================================================
# calc_atom_features
# ===================================================================

class TestCalcAtomFeatures:
    def test_shape_implicit_h(self, ref_ligand_mol):
        af = calc_atom_features(
            ref_ligand_mol.GetAtomWithIdx(0), explicit_H=False,
        )
        assert af.shape == (38,)

    def test_shape_explicit_h(self, ref_ligand_mol):
        af = calc_atom_features(
            ref_ligand_mol.GetAtomWithIdx(0), explicit_H=True,
        )
        assert af.shape == (33,)

    def test_reference_values_first_atom(self, ref_ligand_mol):
        """Pin first-atom features against the reference ligand."""
        af = calc_atom_features(
            ref_ligand_mol.GetAtomWithIdx(0), explicit_H=False,
        )
        expected = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 0,
        ]
        np.testing.assert_array_equal(af, expected)

    def test_all_atoms_have_consistent_shape(self, ref_ligand_mol):
        shapes = set()
        for a in ref_ligand_mol.GetAtoms():
            shapes.add(
                calc_atom_features(a, explicit_H=False).shape[0]
            )
        assert shapes == {38}


# ===================================================================
# calc_bond_features
# ===================================================================

class TestCalcBondFeatures:
    def test_shape_with_chirality(self, ref_ligand_mol):
        bf = calc_bond_features(
            ref_ligand_mol.GetBondWithIdx(0), use_chirality=True,
        )
        assert bf.shape == (10,)

    def test_shape_without_chirality(self, ref_ligand_mol):
        bf = calc_bond_features(
            ref_ligand_mol.GetBondWithIdx(0), use_chirality=False,
        )
        assert bf.shape == (6,)

    def test_reference_values_first_bond(self, ref_ligand_mol):
        bf = calc_bond_features(
            ref_ligand_mol.GetBondWithIdx(0), use_chirality=True,
        )
        expected = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        np.testing.assert_array_equal(bf, expected)

    def test_aromatic_bond_detection(self, benzene_mol):
        """At least one bond in benzene should be aromatic."""
        aromatic_flags = []
        for i in range(benzene_mol.GetNumBonds()):
            bf = calc_bond_features(
                benzene_mol.GetBondWithIdx(i), use_chirality=True,
            )
            aromatic_flags.append(bf[3])
        assert any(f == 1 for f in aromatic_flags)


# ===================================================================
# load_mol
# ===================================================================

class TestLoadMol:
    def test_load_sdf(self, ref_ligand_sdf_path):
        mol = load_mol(
            ref_ligand_sdf_path,
            explicit_H=False,
            use_chirality=True,
        )
        assert mol is not None
        assert mol.GetNumAtoms() == 20
        assert mol.GetNumBonds() == 23

    def test_load_pdb(self, pocket_pdb_path):
        mol = load_mol(
            pocket_pdb_path,
            explicit_H=False,
            use_chirality=False,
        )
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "test.xyz"
        fake.write_text("fake")
        with pytest.raises(IOError, match="only the molecule files"):
            load_mol(str(fake))


# ===================================================================
# mol_to_graph  (ligand graph)
# ===================================================================

class TestMolToGraph:
    def test_reference_ligand_topology(self, ref_ligand_graph):
        """Pin node/edge counts for the reference ligand."""
        assert ref_ligand_graph.num_nodes() == 20
        assert ref_ligand_graph.num_edges() == 46

    def test_atom_feature_shape(self, ref_ligand_graph):
        assert ref_ligand_graph.ndata["atom"].shape == (20, 41)

    def test_bond_feature_shape(self, ref_ligand_graph):
        assert ref_ligand_graph.edata["bond"].shape == (46, 10)

    def test_position_shape(self, ref_ligand_graph):
        assert ref_ligand_graph.ndata["pos"].shape == (20, 3)

    def test_edges_are_bidirectional(self, ref_ligand_graph):
        """Every edge (u,v) should have a reverse (v,u)."""
        src, dst = ref_ligand_graph.edges()
        edge_set = set(zip(src.tolist(), dst.tolist()))
        for s, d in zip(src.tolist(), dst.tolist()):
            assert (d, s) in edge_set

    def test_num_edges_equals_twice_num_bonds(self, ref_ligand_mol):
        g = mol_to_graph(
            ref_ligand_mol, explicit_H=False, use_chirality=True,
        )
        assert g.num_edges() == 2 * ref_ligand_mol.GetNumBonds()

    def test_chirality_adds_3_features(self, ref_ligand_mol):
        g_chir = mol_to_graph(
            ref_ligand_mol, explicit_H=False, use_chirality=True,
        )
        g_nochir = mol_to_graph(
            ref_ligand_mol, explicit_H=False, use_chirality=False,
        )
        assert (
            g_chir.ndata["atom"].shape[1]
            == g_nochir.ndata["atom"].shape[1] + 3
        )

    def test_small_molecule(self, ethanol_mol):
        """Ethanol: 3 heavy atoms, 2 bonds → 4 edges."""
        g = mol_to_graph(
            ethanol_mol, explicit_H=False, use_chirality=True,
        )
        assert g.num_nodes() == 3
        assert g.num_edges() == 4

    def test_positions_finite(self, ref_ligand_graph):
        assert th.isfinite(ref_ligand_graph.ndata["pos"]).all()

    def test_atom_features_finite(self, ref_ligand_graph):
        assert th.isfinite(ref_ligand_graph.ndata["atom"]).all()

    def test_bond_features_are_integer(self, ref_ligand_graph):
        """Bond features are 0/1 indicators."""
        bond = ref_ligand_graph.edata["bond"]
        assert ((bond == 0) | (bond == 1)).all()


# ===================================================================
# prot_to_graph  (protein / pocket graph)
# ===================================================================

class TestProtToGraph:
    def test_reference_pocket_topology(self, pocket_graph):
        """Pin node/edge counts for 1qkt pocket at cutoff 10."""
        assert pocket_graph.num_nodes() == 79
        assert pocket_graph.num_edges() == 2418

    def test_node_feature_shape(self, pocket_graph):
        assert pocket_graph.ndata["feats"].shape == (79, 41)

    def test_edge_feature_shape(self, pocket_graph):
        assert pocket_graph.edata["feats"].shape == (2418, 5)

    def test_position_shape(self, pocket_graph):
        """Each residue has padded atom positions [24, 3]."""
        assert pocket_graph.ndata["pos"].shape == (79, 24, 3)

    def test_node_feature_dim_is_41(self, pocket_graph):
        assert pocket_graph.ndata["feats"].shape[1] == 41

    def test_edge_feature_dim_is_5(self, pocket_graph):
        assert pocket_graph.edata["feats"].shape[1] == 5

    def test_edge_features_finite(self, pocket_graph):
        assert th.isfinite(pocket_graph.edata["feats"]).all()

    def test_node_features_finite(self, pocket_graph):
        assert th.isfinite(pocket_graph.ndata["feats"]).all()

    def test_smaller_cutoff_fewer_edges(self, pocket_pdb_path):
        g5 = prot_to_graph(pocket_pdb_path, 5.0)
        g10 = prot_to_graph(pocket_pdb_path, 10.0)
        assert g5.num_edges() < g10.num_edges()
        assert g5.num_nodes() == g10.num_nodes()
