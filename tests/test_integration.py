"""End-to-end integration tests with pinned reference values.

These tests load the trained model and run full inference on the
example data, asserting that scores match previously captured values.
This is the primary regression guard for dependency upgrades
(DGL → PyG, torch version bumps, numpy, MDAnalysis, etc.).
"""

import numpy as np
import pytest
import torch as th

from RTMScore.model.utils import run_an_eval_epoch

# Reference values captured from the working code.
# Reference values captured from original code (Python 3.8, torch 1.9,
# dgl 0.7, torch-scatter 2.0.9, MDAnalysis 2.0.0) running in Docker.
E2E_PREDS_FIRST10 = [
    38.18545594382491,
    43.82135482018754,
    34.89716250160854,
    60.47760669915718,
    48.603192663962744,
    45.56183402727458,
    36.896138224140834,
    22.76133762041105,
    27.530554150891476,
    56.518945516883025,
]
E2E_PREDS_MIN = 10.682553157427003
E2E_PREDS_MAX = 71.65540131233362
E2E_PREDS_MEAN = 27.47509152326316

EVAL_TOTAL_LOSS = 2.583536144717972
EVAL_MDN_LOSS = 2.5835209225892766
EVAL_ATOM_LOSS = 0.006999623030424118
EVAL_BOND_LOSS = 0.008222504518926144


# ===================================================================
# End-to-end scoring
# ===================================================================

class TestEndToEndScoring:
    def test_prediction_count(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        preds = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert preds.shape == (61,)

    def test_prediction_values_first10(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        preds = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        np.testing.assert_allclose(
            preds[:10], E2E_PREDS_FIRST10, rtol=1e-4,
        )

    def test_prediction_range(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        preds = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        np.testing.assert_allclose(
            preds.min(), E2E_PREDS_MIN, rtol=1e-4,
        )
        np.testing.assert_allclose(
            preds.max(), E2E_PREDS_MAX, rtol=1e-4,
        )

    def test_prediction_mean(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        preds = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        np.testing.assert_allclose(
            preds.mean(), E2E_PREDS_MEAN, rtol=1e-4,
        )


# ===================================================================
# End-to-end loss computation
# ===================================================================

class TestEndToEndLoss:
    def test_eval_losses(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        total, mdn, atom, bond = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=False,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        np.testing.assert_allclose(total, EVAL_TOTAL_LOSS, rtol=1e-4)
        np.testing.assert_allclose(mdn, EVAL_MDN_LOSS, rtol=1e-4)
        np.testing.assert_allclose(atom, EVAL_ATOM_LOSS, rtol=1e-3)
        np.testing.assert_allclose(bond, EVAL_BOND_LOSS, rtol=1e-3)


# ===================================================================
# Atom and residue contribution decomposition
# ===================================================================

class TestContributions:
    def test_atom_contributions(
        self, trained_model, full_dataloader, model_kwargs,
        vs_dataset,
    ):
        preds, at_contrs, _ = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            atom_contribution=True,
            res_contribution=False,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert preds.shape == (61,)
        assert len(at_contrs) == 61
        # First ligand atom count
        first_lig_nodes = vs_dataset[0][1].num_nodes()
        assert at_contrs[0].shape == (first_lig_nodes,)

    def test_residue_contributions(
        self, trained_model, full_dataloader, model_kwargs,
        vs_dataset,
    ):
        preds, _, res_contrs = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            atom_contribution=False,
            res_contribution=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert preds.shape == (61,)
        assert len(res_contrs) == 61
        # Pocket residue count
        prot_nodes = vs_dataset[0][2].num_nodes()
        assert res_contrs[0].shape == (prot_nodes,)

    def test_atom_contribs_non_negative(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        _, at_contrs, _ = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            atom_contribution=True,
            res_contribution=False,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        for ac in at_contrs:
            assert (ac >= 0).all()

    def test_residue_contribs_non_negative(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        _, _, res_contrs = run_an_eval_epoch(
            trained_model,
            full_dataloader,
            pred=True,
            atom_contribution=False,
            res_contribution=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        for rc in res_contrs:
            assert (rc >= 0).all()


# ===================================================================
# Model checkpoint loading
# ===================================================================

class TestCheckpointLoading:
    def test_all_three_models_loadable(self, model_kwargs):
        """All three shipped .pth files should load without error."""
        import os
        from RTMScore.model.model2 import (
            DGLGraphTransformer,
            RTMScore,
        )

        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trained_models",
        )
        for name in [
            "rtmscore_model1.pth",
            "rtmscore_model2.pth",
            "rtmscore_model3.pth",
        ]:
            path = os.path.join(models_dir, name)
            if not os.path.exists(path):
                pytest.skip(f"{name} not found")

            lm = DGLGraphTransformer(
                in_channels=41, edge_features=10,
                num_hidden_channels=128,
                activ_fn=th.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply="batch",
                dropout_rate=0.15,
                num_layers=6,
            )
            pm = DGLGraphTransformer(
                in_channels=41, edge_features=5,
                num_hidden_channels=128,
                activ_fn=th.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply="batch",
                dropout_rate=0.15,
                num_layers=6,
            )
            model = RTMScore(
                lm, pm,
                in_channels=128,
                hidden_dim=128,
                n_gaussians=10,
                dropout_rate=0.10,
                dist_threhold=5,
            )
            ckpt = th.load(
                path,
                map_location="cpu",
            )
            model.load_state_dict(ckpt["model_state_dict"])

    def test_loaded_model_deterministic(
        self, trained_model, full_dataloader, model_kwargs,
    ):
        """Two consecutive eval runs produce identical results."""
        p1 = run_an_eval_epoch(
            trained_model, full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        p2 = run_an_eval_epoch(
            trained_model, full_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        np.testing.assert_array_equal(p1, p2)
