"""Tests for RTMScore.model.model2.

Covers the GraphTransformer encoder, the full RTMScore model
forward pass, the to_dense_batch_pyg helper, and parameter counts.
These are critical regression guards for a DGL -> PyG migration.
"""

import numpy as np
import pytest
import torch as th
from torch_geometric.data import Data, Batch

from RTMScore.model.model2 import (
    GraphTransformer,
    GraphTransformerModule,
    FinalGraphTransformerModule,
    MultiHeadAttentionLayer,
    RTMScore,
    to_dense_batch_pyg,
)


# ===================================================================
# GraphTransformer
# ===================================================================

class TestGraphTransformer:
    def test_output_shape(self, ref_ligand_graph):
        """Forward pass returns [N, hidden_dim] node features."""
        model = GraphTransformer(
            in_channels=41,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=th.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply="batch",
            dropout_rate=0.15,
            num_layers=6,
        )
        model.eval()
        g = ref_ligand_graph
        with th.no_grad():
            out = model(g, g.atom.float(), g.bond.float())
        assert out.shape == (g.num_nodes, 128)

    def test_single_layer(self):
        """num_layers=1 produces only a FinalGraphTransformerModule."""
        model = GraphTransformer(
            in_channels=10,
            edge_features=5,
            num_hidden_channels=32,
            num_layers=1,
        )
        assert len(model.gt_block) == 1
        assert isinstance(model.gt_block[0], FinalGraphTransformerModule)

    def test_multi_layer_structure(self):
        model = GraphTransformer(
            in_channels=10,
            edge_features=5,
            num_hidden_channels=32,
            num_layers=4,
        )
        assert len(model.gt_block) == 4
        for layer in model.gt_block[:-1]:
            assert isinstance(layer, GraphTransformerModule)
        assert isinstance(
            model.gt_block[-1], FinalGraphTransformerModule,
        )

    def test_batch_norm_vs_layer_norm(self):
        """Both normalization modes should produce valid output."""
        for norm in ("batch", "layer"):
            model = GraphTransformer(
                in_channels=10,
                edge_features=5,
                num_hidden_channels=32,
                num_layers=2,
                norm_to_apply=norm,
            )
            g = Data(
                edge_index=th.tensor([[0, 1], [1, 0]], dtype=th.long),
                num_nodes=2,
            )
            node_feats = th.randn(2, 10)
            edge_feats = th.randn(2, 5)
            model.eval()
            with th.no_grad():
                out = model(g, node_feats, edge_feats)
            assert out.shape == (2, 32)
            assert th.isfinite(out).all()


# ===================================================================
# MultiHeadAttentionLayer
# ===================================================================

class TestMultiHeadAttentionLayer:
    def test_output_shapes(self):
        layer = MultiHeadAttentionLayer(
            num_input_feats=32,
            num_output_feats=8,
            num_heads=4,
            using_bias=False,
            update_edge_feats=True,
        )
        edge_index = th.tensor([[0, 1, 2], [1, 2, 0]], dtype=th.long)
        node_feats = th.randn(3, 32)
        edge_feats = th.randn(3, 32)
        h_out, e_out = layer(edge_index, node_feats, edge_feats, num_nodes=3)
        assert h_out.shape == (3, 4, 8)
        assert e_out.shape == (3, 4, 8)

    def test_no_edge_update(self):
        layer = MultiHeadAttentionLayer(
            num_input_feats=32,
            num_output_feats=8,
            num_heads=4,
            using_bias=False,
            update_edge_feats=False,
        )
        edge_index = th.tensor([[0, 1], [1, 0]], dtype=th.long)
        h_out, e_out = layer(
            edge_index, th.randn(2, 32), th.randn(2, 32), num_nodes=2,
        )
        assert h_out.shape == (2, 4, 8)
        assert e_out is None


# ===================================================================
# to_dense_batch_pyg
# ===================================================================

class TestToDenseBatchDgl:
    def test_single_graph(self):
        g = Data(
            edge_index=th.tensor([[0, 1], [1, 0]], dtype=th.long),
            num_nodes=3,
        )
        # For to_dense_batch we need a batch attribute
        bg = Batch.from_data_list([g])
        feats = th.tensor([[1.0], [2.0], [3.0]])
        out, mask = to_dense_batch_pyg(bg, feats)
        assert out.shape == (1, 3, 1)
        assert mask.shape == (1, 3)
        assert mask.all()

    def test_batched_graphs_padding(self):
        g1 = Data(
            edge_index=th.tensor([[0], [1]], dtype=th.long),
            num_nodes=2,
        )
        g2 = Data(
            edge_index=th.tensor([[0, 1, 2], [1, 2, 0]], dtype=th.long),
            num_nodes=4,
        )
        bg = Batch.from_data_list([g1, g2])
        feats = th.randn(6, 8)
        out, mask = to_dense_batch_pyg(bg, feats)
        assert out.shape == (2, 4, 8)
        assert mask.shape == (2, 4)
        assert mask[0, :2].all()
        assert not mask[0, 2:].any()
        assert mask[1, :4].all()

    def test_fill_value(self):
        g1 = Data(
            edge_index=th.tensor([[0], [1]], dtype=th.long),
            num_nodes=2,
        )
        g2 = Data(
            edge_index=th.zeros(2, 0, dtype=th.long),
            num_nodes=3,
        )
        bg = Batch.from_data_list([g1, g2])
        feats = th.ones(5, 1)
        out, _ = to_dense_batch_pyg(bg, feats, fill_value=-1)
        assert (out[0, 2:] == -1).all()


# ===================================================================
# RTMScore model
# ===================================================================

class TestRTMScoreModel:
    def test_parameter_counts(self, untrained_model):
        total = sum(p.numel() for p in untrained_model.parameters())
        assert total == 2656435

    def test_lig_encoder_params(self, untrained_model):
        lig_params = sum(
            p.numel()
            for p in untrained_model.lig_model.parameters()
        )
        assert lig_params == 1308416

    def test_prot_encoder_params(self, untrained_model):
        prot_params = sum(
            p.numel()
            for p in untrained_model.prot_model.parameters()
        )
        assert prot_params == 1307776

    def test_forward_output_shapes(
        self, untrained_model, vs_dataset,
    ):
        """Forward pass on a 3-sample batch returns correct shapes."""
        from RTMScore.model.utils import collate, set_random_seed

        set_random_seed(42)
        batch = [vs_dataset[i] for i in range(3)]
        _, bgl, bgp = collate(batch)
        untrained_model.eval()
        with th.no_grad():
            pi, sigma, mu, dist, at, bt, cb = untrained_model(
                bgp, bgl,
            )

        n_pairs = pi.shape[0]
        assert pi.shape == (n_pairs, 10)
        assert sigma.shape == (n_pairs, 10)
        assert mu.shape == (n_pairs, 10)
        assert dist.shape == (n_pairs, 1)
        # atom_types: [total_lig_atoms, 17]
        total_lig_atoms = bgl.num_nodes
        assert at.shape == (total_lig_atoms, 17)
        # bond_types: [total_lig_edges, 4]
        total_lig_edges = bgl.num_edges
        assert bt.shape == (total_lig_edges, 4)

    def test_forward_reference_shapes_3_samples(
        self, untrained_model, vs_dataset,
    ):
        """Pin the exact pair count for the first 3 decoys."""
        from RTMScore.model.utils import collate, set_random_seed

        set_random_seed(42)
        batch = [vs_dataset[i] for i in range(3)]
        _, bgl, bgp = collate(batch)
        untrained_model.eval()
        with th.no_grad():
            pi, sigma, mu, dist, at, bt, cb = untrained_model(
                bgp, bgl,
            )
        assert pi.shape == (4740, 10)
        assert at.shape == (60, 17)
        assert bt.shape == (138, 4)

    def test_dist_threshold_stored(self, untrained_model):
        assert untrained_model.dist_threhold == 5

    def test_n_gaussians_pi_output(self, model_kwargs):
        """Changing n_gaussians changes the MDN output dimension."""
        from RTMScore.model.model2 import (
            GraphTransformer,
            RTMScore,
        )

        for ng in (5, 15):
            lm = GraphTransformer(
                in_channels=41, edge_features=10,
                num_hidden_channels=128, num_layers=2,
            )
            pm = GraphTransformer(
                in_channels=41, edge_features=5,
                num_hidden_channels=128, num_layers=2,
            )
            m = RTMScore(lm, pm, 128, 128, ng, 0.1, 5)
            assert m.z_pi.out_features == ng
            assert m.z_sigma.out_features == ng
            assert m.z_mu.out_features == ng
