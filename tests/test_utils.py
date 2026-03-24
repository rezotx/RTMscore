"""Tests for RTMScore.model.utils.

Covers the collate function, MDN loss, probability calculation,
Meter metrics, EarlyStopping, and the train/eval epoch runners.
"""

import os
import tempfile

import numpy as np
import pytest
import torch as th
from torch_geometric.data import Data, Batch

from RTMScore.model.utils import (
    EarlyStopping,
    Meter,
    calculate_probablity,
    collate,
    mdn_loss_fn,
    run_a_train_epoch,
    run_an_eval_epoch,
    set_random_seed,
)


# ===================================================================
# collate
# ===================================================================

class TestCollate:
    def test_batch_structure(self, vs_dataset):
        items = [vs_dataset[i] for i in range(3)]
        pdbids, bgl, bgp = collate(items)
        assert len(pdbids) == 3
        assert bgl.num_graphs == 3
        assert bgp.num_graphs == 3

    def test_node_counts_sum(self, vs_dataset):
        items = [vs_dataset[i] for i in range(3)]
        individual_lig_nodes = [
            vs_dataset[i][1].num_nodes for i in range(3)
        ]
        _, bgl, _ = collate(items)
        assert bgl.num_nodes == sum(individual_lig_nodes)

    def test_single_item(self, vs_dataset):
        items = [vs_dataset[0]]
        pdbids, bgl, bgp = collate(items)
        assert len(pdbids) == 1
        assert bgl.num_graphs == 1


# ===================================================================
# mdn_loss_fn
# ===================================================================

class TestMdnLoss:
    def test_output_shape(self):
        n, k = 100, 10
        pi = th.softmax(th.randn(n, k), dim=-1)
        sigma = th.abs(th.randn(n, k)) + 1.1
        mu = th.abs(th.randn(n, k)) + 1.0
        y = th.randn(n, 1)
        loss = mdn_loss_fn(pi, sigma, mu, y)
        assert loss.shape == (n,)

    def test_loss_is_finite(self):
        n, k = 50, 5
        pi = th.softmax(th.randn(n, k), dim=-1)
        sigma = th.abs(th.randn(n, k)) + 1.1
        mu = th.abs(th.randn(n, k)) + 1.0
        y = th.randn(n, 1)
        loss = mdn_loss_fn(pi, sigma, mu, y)
        assert th.isfinite(loss).all()

    def test_loss_positive_on_typical_inputs(self):
        """MDN NLL should be positive for typical distance inputs."""
        n, k = 100, 10
        pi = th.softmax(th.randn(n, k), dim=-1)
        sigma = th.ones(n, k) * 2.0
        mu = th.ones(n, k) * 3.0
        y = th.ones(n, 1) * 5.0
        loss = mdn_loss_fn(pi, sigma, mu, y)
        assert (loss > 0).all()


# ===================================================================
# calculate_probablity
# ===================================================================

class TestCalculateProbability:
    def test_output_shape(self):
        n, k = 100, 10
        pi = th.softmax(th.randn(n, k), dim=-1)
        sigma = th.abs(th.randn(n, k)) + 1.1
        mu = th.abs(th.randn(n, k)) + 1.0
        y = th.randn(n, 1)
        prob = calculate_probablity(pi, sigma, mu, y)
        assert prob.shape == (n,)

    def test_probabilities_non_negative(self):
        n, k = 50, 5
        pi = th.softmax(th.randn(n, k), dim=-1)
        sigma = th.abs(th.randn(n, k)) + 1.1
        mu = th.abs(th.randn(n, k)) + 1.0
        y = th.randn(n, 1)
        prob = calculate_probablity(pi, sigma, mu, y)
        assert (prob >= 0).all()


# ===================================================================
# Meter
# ===================================================================

class TestMeter:
    @pytest.fixture
    def populated_meter(self):
        m = Meter()
        m.update(
            th.tensor([[0.8, 0.2]]),
            th.tensor([[1.0, 0.0]]),
        )
        m.update(
            th.tensor([[0.3, 0.7]]),
            th.tensor([[0.0, 1.0]]),
        )
        return m

    def test_pearson_r(self, populated_meter):
        rp = populated_meter.pearson_r()
        assert len(rp) == 2
        assert all(abs(r - 1.0) < 1e-6 for r in rp)

    def test_spearman_r(self, populated_meter):
        rs = populated_meter.spearman_r()
        assert len(rs) == 2
        assert all(abs(r - 1.0) < 1e-6 for r in rs)

    def test_mae(self, populated_meter):
        mae = populated_meter.mae()
        assert len(mae) == 2
        np.testing.assert_allclose(mae, [0.25, 0.25], atol=1e-6)

    def test_rmse(self, populated_meter):
        rmse = populated_meter.rmse()
        assert len(rmse) == 2
        np.testing.assert_allclose(
            rmse, [0.2549509, 0.2549510], atol=1e-4,
        )

    def test_compute_metric_rp(self, populated_meter):
        rp = populated_meter.compute_metric("rp")
        assert len(rp) == 2

    def test_compute_metric_invalid_raises(self, populated_meter):
        with pytest.raises(ValueError, match="Expect metric_name"):
            populated_meter.compute_metric("invalid")

    def test_reduction_mean(self, populated_meter):
        mae_mean = populated_meter.mae(reduction="mean")
        assert isinstance(mae_mean, float)
        np.testing.assert_allclose(mae_mean, 0.25, atol=1e-6)

    def test_reduction_sum(self, populated_meter):
        mae_sum = populated_meter.mae(reduction="sum")
        np.testing.assert_allclose(mae_sum, 0.50, atol=1e-6)

    def test_with_normalization(self):
        m = Meter(
            mean=th.tensor([0.5, 0.5]),
            std=th.tensor([2.0, 2.0]),
        )
        m.update(
            th.tensor([[0.1, 0.2]]),
            th.tensor([[0.3, 0.4]]),
        )
        mask, y_pred, y_true = m._finalize()
        assert y_pred.shape == (1, 2)
        expected_pred = th.tensor([[0.1, 0.2]]) * 2.0 + 0.5
        th.testing.assert_close(y_pred, expected_pred)


# ===================================================================
# EarlyStopping
# ===================================================================

class TestEarlyStopping:
    def test_higher_mode(self, untrained_model):
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            es = EarlyStopping(
                mode="higher", patience=3, filename=f.name,
            )
            assert not es.step(0.5, untrained_model)  # best=0.5
            assert not es.step(0.6, untrained_model)  # best=0.6
            assert not es.step(0.4, untrained_model)  # counter=1
            assert not es.step(0.3, untrained_model)  # counter=2
            assert es.step(0.2, untrained_model)       # counter=3

    def test_lower_mode(self, untrained_model):
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            es = EarlyStopping(
                mode="lower", patience=2, filename=f.name,
            )
            assert not es.step(1.0, untrained_model)  # best=1.0
            assert not es.step(0.9, untrained_model)  # best=0.9
            assert not es.step(1.0, untrained_model)  # counter=1
            assert es.step(1.1, untrained_model)       # counter=2

    def test_checkpoint_save_load(self, untrained_model):
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            es = EarlyStopping(
                mode="lower", patience=5, filename=f.name,
            )
            es.step(1.0, untrained_model)
            # Modify a parameter
            with th.no_grad():
                untrained_model.z_pi.weight.fill_(999.0)
            # Load checkpoint should restore original weights
            es.load_checkpoint(untrained_model)
            assert not (
                untrained_model.z_pi.weight == 999.0
            ).all()


# ===================================================================
# set_random_seed
# ===================================================================

class TestSetRandomSeed:
    def test_reproducibility(self):
        set_random_seed(42)
        a = th.randn(10)
        set_random_seed(42)
        b = th.randn(10)
        th.testing.assert_close(a, b)


# ===================================================================
# run_an_eval_epoch  (prediction mode)
# ===================================================================

class TestRunAnEvalEpoch:
    def test_pred_returns_numpy(
        self, trained_model, small_dataloader, model_kwargs,
    ):
        preds = run_an_eval_epoch(
            trained_model,
            small_dataloader,
            pred=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1

    def test_loss_returns_four_floats(
        self, trained_model, small_dataloader, model_kwargs,
    ):
        result = run_an_eval_epoch(
            trained_model,
            small_dataloader,
            pred=False,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        total, mdn, atom, bond = result
        assert isinstance(total, float)
        assert isinstance(mdn, float)
        assert np.isfinite(total)
        assert np.isfinite(mdn)

    def test_atom_contribution_shapes(
        self, trained_model, small_dataloader, model_kwargs,
    ):
        preds, at_contrs, res_contrs = run_an_eval_epoch(
            trained_model,
            small_dataloader,
            pred=True,
            atom_contribution=True,
            res_contribution=False,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert preds.shape[0] == 5
        assert len(at_contrs) == 5
        assert len(res_contrs) == 0
        for ac in at_contrs:
            assert isinstance(ac, np.ndarray)
            assert ac.ndim == 1

    def test_residue_contribution_shapes(
        self, trained_model, small_dataloader, model_kwargs,
    ):
        preds, at_contrs, res_contrs = run_an_eval_epoch(
            trained_model,
            small_dataloader,
            pred=True,
            atom_contribution=False,
            res_contribution=True,
            dist_threhold=model_kwargs["dist_threhold"],
            device="cpu",
        )
        assert preds.shape[0] == 5
        assert len(at_contrs) == 0
        assert len(res_contrs) == 5
        for rc in res_contrs:
            assert isinstance(rc, np.ndarray)
            assert rc.ndim == 1


# ===================================================================
# run_a_train_epoch  (single training step)
# ===================================================================

class TestRunATrainEpoch:
    def test_single_epoch_returns_losses(
        self, untrained_model, small_dataloader,
    ):
        optimizer = th.optim.Adam(
            untrained_model.parameters(), lr=1e-4,
        )
        total, mdn, atom, bond = run_a_train_epoch(
            epoch=0,
            model=untrained_model,
            data_loader=small_dataloader,
            optimizer=optimizer,
            aux_weight=0.001,
            device="cpu",
        )
        assert isinstance(total, float)
        assert np.isfinite(total)
        assert np.isfinite(mdn)
