"""Batched-energy contract for DeviceProperty.

The recursive Green's function kernel was widened to take a leading energy-batch
dim; this test pins that DeviceProperty.cal_green_function, _cal_tc_,
_cal_dos_, and _cal_ldos_ produce the same numbers whether the caller drives B
scalar energies in a loop or feeds a 1-D [B] energy tensor in one shot.

It also pins the runner-style chunking behavior (split [E] into chunks of size
e_batch_size, cat the per-chunk outputs back together).
"""

from unittest.mock import Mock

import pytest
import torch

from dpnegf.negf.device_property import DeviceProperty

from .test_recursive_gf_batched import _make_btd_inputs


class _FakeHamiltonian:
    """Minimal hamiltonian: cal_green_function calls get_hs_device(...) to
    fill self.hd/sd/hl/su/sl/hu; we just hand back pre-built BTD blocks."""

    def __init__(self, hd, sd, hl, hu, sl, su, device_norbs):
        self._packed = (hd, sd, hl, su, sl, hu)
        self.device_norbs = device_norbs

    def get_hs_device(self, kpoint, V, block_tridiagonal, device=None):
        assert device is not None
        return tuple([block.to(device) for block in parts] for parts in self._packed)


class _FakeLead:
    """Per-E scalar se store + a Fermi-Dirac at chemiPot=0, kBT=0.025 (~300K)."""

    def __init__(self):
        self.se = None
        self.chemiPot_lead = 0.0
        self.kBT = 0.025

    def fermi_dirac(self, x):
        return 1.0 / (1.0 + torch.exp((x - self.chemiPot_lead) / self.kBT))

    @property
    def gamma(self):
        # Mirrors LeadProperty.gamma after the .mH fix — shape-agnostic.
        return 1j * (self.se - self.se.mH)


def _make_deviceprop(block_sizes, hd, sd, hl, hu, sl, su):
    """Construct a DeviceProperty with the fakes wired in."""
    n_total = sum(block_sizes)
    norbs_per_atom = [1] * n_total  # one orbital per "atom" so ldos == per-orb DOS
    ham = _FakeHamiltonian(hd, sd, hl, hu, sl, su, device_norbs=norbs_per_atom)

    dev = DeviceProperty.__new__(DeviceProperty)
    dev.greenfuncs = 0
    dev.hamiltonian = ham
    dev.structure = None
    dev.results_path = "/tmp"
    dev.cdtype = torch.complex128
    dev.rgf_device = "cpu"
    dev.kBT = 0.025
    dev.e_T = 300.0
    dev.chemiPot = None
    dev.E_ref = 0.0
    dev.kpoint = None
    dev.newK_flag = None
    dev.newV_flag = None
    dev.lead_L = _FakeLead()
    dev.lead_R = _FakeLead()
    return dev


def _block_sizes_match_se(left_se_2d, right_se_2d, block_sizes):
    # cal_green_function asserts se shape <= corresponding hd block; with the BTD
    # builder, block_sizes[0] == left_se size and block_sizes[-1] == right_se size.
    assert left_se_2d.shape == (block_sizes[0], block_sizes[0])
    assert right_se_2d.shape == (block_sizes[-1], block_sizes[-1])


@pytest.mark.parametrize("change", ["kpoint", "bias", "layout"])
def test_prepared_hamiltonian_is_reused_until_context_changes(tmp_path, change):
    sizes = [3, 4, 2]
    hd, sd, hl, hu, sl, su, se_l, se_r, energies = _make_btd_inputs(1, sizes, seed=11)
    dev = _make_deviceprop(sizes, hd, sd, hl, hu, sl, su)
    dev.results_path = str(tmp_path)
    original_load = dev.hamiltonian.get_hs_device
    load = Mock(wraps=original_load)
    dev.hamiltonian.get_hs_device = load
    k, bias = [0., 0., 0.], torch.tensor(0.1)

    dev.prepare_hamiltonian(k, True, bias)
    resident = dev.hd
    dev.lead_L.se, dev.lead_R.se = se_l[0], se_r[0]
    dev.cal_green_function(energies[0], k, Vbias=bias, need_trans_eigs=True)
    green, eigenstate = dev.greenfuncs, dev._trans_eigs_state
    assert green and eigenstate is not None
    dev.prepare_hamiltonian(k, True, bias)
    assert load.call_count == 1
    assert dev.hd is resident
    assert dev.greenfuncs is green
    assert dev._trans_eigs_state is eigenstate

    def load_changed(*args, **kwargs):
        assert dev.greenfuncs == 0
        assert dev._trans_eigs_state is None
        return original_load(*args, **kwargs)

    load.side_effect = load_changed

    if change == "kpoint":
        k = [0.5, 0., 0.]
    elif change == "bias":
        bias = torch.tensor(0.2)
    dev.prepare_hamiltonian(k, change != "layout", bias)
    assert load.call_count == 2
    assert dev.hd is not resident
    assert dev.greenfuncs == 0
    assert dev._trans_eigs_state is None
    with pytest.raises(RuntimeError, match="Compute full RGF"):
        dev.cal_transmission_eigs()


@pytest.fixture
def solved_device(tmp_path):
    sizes = [3, 4, 2]
    hd, sd, hl, hu, sl, su, se_l, se_r, energies = _make_btd_inputs(1, sizes, seed=11)
    dev = _make_deviceprop(sizes, hd, sd, hl, hu, sl, su)
    dev.results_path = str(tmp_path)
    dev.lead_L.se, dev.lead_R.se = se_l[0], se_r[0]
    dev.cal_green_function(energies[0], [0., 0., 0.], Vbias=torch.tensor(0.1),
                           need_trans_eigs=True)
    return dev, energies[0]


def _inject_hs_failure(dev, stage, error):
    original_load = dev.hamiltonian.get_hs_device

    def load(*args, **kwargs):
        # Derived state must already be invalid when a replacement load starts.
        assert dev.greenfuncs == 0
        assert dev._trans_eigs_state is None
        if stage == "load":
            raise error
        parts = tuple([block.clone() for block in blocks]
                      for blocks in original_load(*args, **kwargs))
        if stage == "transfer":
            parts[-1][0] = Mock()
            parts[-1][0].to.side_effect = error
        return parts

    if stage == "mapping":
        dev.hamiltonian.get_orbital_map = Mock(side_effect=error)
    dev.hamiltonian.get_hs_device = Mock(side_effect=load)
    return original_load


@pytest.mark.parametrize("change", ["kpoint", "bias", "layout"])
@pytest.mark.parametrize("stage", ["load", "transfer", "mapping"])
def test_failed_hamiltonian_switch_preserves_committed_context(solved_device, change, stage):
    dev, _ = solved_device
    names = ("hd", "sd", "hl", "su", "sl", "hu", "_orbital_mapping_snapshot",
             "kpoint", "V", "oldV", "block_tridiagonal", "newK_flag", "newV_flag")
    previous = {name: getattr(dev, name) for name in names}
    k = [0.5, 0., 0.] if change == "kpoint" else [0., 0., 0.]
    bias = torch.tensor(0.2 if change == "bias" else 0.1)
    layout = change != "layout"
    error = RuntimeError(f"H/S {stage} failed")
    original_load = _inject_hs_failure(dev, stage, error)

    with pytest.raises(RuntimeError, match=f"H/S {stage} failed") as caught:
        dev.prepare_hamiltonian(k, layout, bias)
    assert caught.value is error
    for name, value in previous.items():
        assert getattr(dev, name) is value, name
    assert dev.greenfuncs == 0
    assert dev._trans_eigs_state is None
    with pytest.raises(RuntimeError, match="Compute full RGF"):
        dev.cal_transmission_eigs()

    # The failed request must not be mistaken for a resident cache hit on retry.
    load = dev.hamiltonian.get_hs_device
    load.side_effect = original_load
    if stage == "mapping":
        del dev.hamiltonian.get_orbital_map
    dev.prepare_hamiltonian(k, layout, bias)
    assert load.call_count == 2
    assert dev.hd is not previous["hd"]
    torch.testing.assert_close(dev.kpoint, torch.tensor(k))
    torch.testing.assert_close(dev.V, bias)
    assert dev.block_tridiagonal == layout
    dev.prepare_hamiltonian(k, layout, bias)
    assert load.call_count == 2


@pytest.mark.parametrize("stage", ["load", "transfer", "mapping"])
def test_initial_hamiltonian_failure_does_not_publish_context(tmp_path, stage):
    sizes = [3, 4, 2]
    hd, sd, hl, hu, sl, su, *_ = _make_btd_inputs(1, sizes, seed=11)
    dev = _make_deviceprop(sizes, hd, sd, hl, hu, sl, su)
    dev.results_path = str(tmp_path)
    error = RuntimeError(f"H/S {stage} failed")
    _inject_hs_failure(dev, stage, error)

    with pytest.raises(RuntimeError, match=f"H/S {stage} failed") as caught:
        dev.prepare_hamiltonian([0.5, 0., 0.], False, torch.tensor(0.2))
    assert caught.value is error
    assert dev.kpoint is None
    assert dev.newK_flag is None
    assert dev.newV_flag is None
    for name in ("hd", "sd", "hl", "su", "sl", "hu", "V", "oldV",
                 "block_tridiagonal", "_orbital_mapping_snapshot"):
        assert not hasattr(dev, name), name
    assert dev.greenfuncs == 0
    assert dev._trans_eigs_state is None


def test_failed_green_replacement_clears_both_derived_states(solved_device):
    dev, energy = solved_device
    resident = dev.hd
    with pytest.raises(ValueError, match="Unknown gf_cal_mode"):
        dev.cal_green_function(energy + 0.1, [0., 0., 0.], Vbias=torch.tensor(0.1),
                               gf_cal_mode="invalid")
    assert dev.hd is resident
    assert dev.greenfuncs == 0
    assert dev._trans_eigs_state is None


@pytest.mark.parametrize("keep_hs", [True, False])
def test_dos_reuses_prepared_context_and_restores_only_released_blocks(solved_device, keep_hs):
    dev, energy = solved_device
    expected = dev._cal_dos_()
    dev.cal_green_function(energy, [0., 0., 0.], Vbias=torch.tensor(0.1),
                           HS_inmem=keep_hs, need_trans_eigs=True)
    green, eigenstate = dev.greenfuncs, dev._trans_eigs_state
    load = Mock(wraps=dev.hamiltonian.get_hs_device)
    dev.hamiltonian.get_hs_device = load
    torch.testing.assert_close(dev._cal_dos_(), expected)
    assert load.call_count == (0 if keep_hs else 1)
    assert dev.greenfuncs is green
    assert dev._trans_eigs_state is eigenstate


def test_context_tags_do_not_alias_mutable_inputs(solved_device):
    dev, _ = solved_device
    k, bias = torch.tensor([0.5, 0., 0.]), torch.tensor(0.2)
    load = Mock(wraps=dev.hamiltonian.get_hs_device)
    dev.hamiltonian.get_hs_device = load
    dev.prepare_hamiltonian(k, True, bias)
    k[0], bias[...] = 0.25, 0.3
    torch.testing.assert_close(dev.kpoint, torch.tensor([0.5, 0., 0.]))
    torch.testing.assert_close(dev.V, torch.tensor(0.2))
    dev.prepare_hamiltonian(k, True, bias)
    assert load.call_count == 2


def _run_scalar(dev, energies_1d, left_se_batched, right_se_batched, need_lesser):
    """Drive cal_green_function B times with scalar energies; collect tc/dos/ldos."""
    tc_list, dos_list, ldos_list, gnd0_list = [], [], [], []
    for b in range(energies_1d.shape[0]):
        dev.lead_L.se = left_se_batched[b]
        dev.lead_R.se = right_se_batched[b]
        dev.cal_green_function(
            energy=energies_1d[b],
            kpoint=[0.0, 0.0, 0.0],
            eta_device=0.0,
            block_tridiagonal=True,
            need_lesser=need_lesser,
            need_gr_lc=False,
        )
        tc_list.append(dev._cal_tc_())
        dos_list.append(dev._cal_dos_())
        ldos_list.append(dev._cal_ldos_())
        if need_lesser:
            gnd0_list.append(dev.greenfuncs["gnd"][0])
    return (
        torch.stack(tc_list),
        torch.stack(dos_list),
        torch.stack(ldos_list),
        torch.stack(gnd0_list) if need_lesser else None,
    )


def _run_batched(dev, energies_1d, left_se_batched, right_se_batched, need_lesser):
    """Drive cal_green_function once with a 1-D [B] energy."""
    dev.lead_L.se = left_se_batched
    dev.lead_R.se = right_se_batched
    dev.cal_green_function(
        energy=energies_1d,
        kpoint=[0.0, 0.0, 0.0],
        eta_device=0.0,
        block_tridiagonal=True,
        need_lesser=need_lesser,
        need_gr_lc=False,
    )
    tc_B = dev._cal_tc_()
    dos_B = dev._cal_dos_()
    ldos_B = dev._cal_ldos_()
    gnd0_B = dev.greenfuncs["gnd"][0] if need_lesser else None
    return tc_B, dos_B, ldos_B, gnd0_B


@pytest.mark.parametrize("need_lesser", [False, True])
def test_batched_matches_scalar(need_lesser):
    B = 7
    block_sizes = [4, 4, 4]
    hd, sd, hl, hu, sl, su, left_se_B, right_se_B, energies_cplx = _make_btd_inputs(
        B, block_sizes, seed=11
    )
    # The runner feeds real-valued energies (the kernel adds 1j*eta itself); strip
    # the seed's tiny imaginary offset to keep this path realistic.
    energies = energies_cplx.real.to(torch.complex128)

    # Hermitize lead self-energies so .mH-based gamma is a real broadening function.
    left_se_B = 0.5 * (left_se_B + left_se_B.mH) + 1j * 0.1 * torch.eye(
        block_sizes[0], dtype=torch.complex128
    )
    right_se_B = 0.5 * (right_se_B + right_se_B.mH) + 1j * 0.1 * torch.eye(
        block_sizes[-1], dtype=torch.complex128
    )
    _block_sizes_match_se(left_se_B[0], right_se_B[0], block_sizes)

    dev = _make_deviceprop(block_sizes, hd, sd, hl, hu, sl, su)

    tc_e, dos_e, ldos_e, gnd0_e = _run_scalar(
        dev, energies, left_se_B, right_se_B, need_lesser
    )

    # Reset cached state so the batched call goes through the same init path.
    dev.kpoint = None
    dev.newK_flag = None
    dev.newV_flag = None
    if hasattr(dev, "V"):
        del dev.V
    if hasattr(dev, "hd"):
        del dev.hd, dev.sd, dev.hl, dev.su, dev.sl, dev.hu

    tc_B, dos_B, ldos_B, gnd0_B = _run_batched(
        dev, energies, left_se_B, right_se_B, need_lesser
    )

    assert tc_B.shape == (B,)
    assert dos_B.shape == (B,)
    assert ldos_B.shape == (B, sum(block_sizes))
    assert torch.allclose(tc_B, tc_e, atol=1e-10, rtol=0)
    assert torch.allclose(dos_B, dos_e, atol=1e-10, rtol=0)
    assert torch.allclose(ldos_B, ldos_e, atol=1e-10, rtol=0)
    if need_lesser:
        assert gnd0_B.shape == (B, block_sizes[0], block_sizes[0])
        assert torch.allclose(gnd0_B, gnd0_e, atol=1e-10, rtol=0)


def test_chunked_matches_unchunked():
    """Runner-style chunking: split [E] into chunks of e_batch_size, cat back."""
    B = 7
    block_sizes = [4, 4, 4]
    hd, sd, hl, hu, sl, su, left_se_B, right_se_B, energies_cplx = _make_btd_inputs(
        B, block_sizes, seed=13
    )
    energies = energies_cplx.real.to(torch.complex128)
    left_se_B = 0.5 * (left_se_B + left_se_B.mH) + 1j * 0.1 * torch.eye(
        block_sizes[0], dtype=torch.complex128
    )
    right_se_B = 0.5 * (right_se_B + right_se_B.mH) + 1j * 0.1 * torch.eye(
        block_sizes[-1], dtype=torch.complex128
    )

    dev = _make_deviceprop(block_sizes, hd, sd, hl, hu, sl, su)

    # Unchunked baseline: single B=7 batched call.
    tc_full, dos_full, ldos_full, _ = _run_batched(
        dev, energies, left_se_B, right_se_B, need_lesser=False
    )

    # Chunked: e_batch_size=3 → splits of [3,3,1].
    e_batch_size = 3
    tc_chunks, dos_chunks, ldos_chunks = [], [], []
    for chunk_idx, e_chunk in enumerate(torch.split(energies, e_batch_size)):
        b = len(e_chunk)
        i0 = chunk_idx * e_batch_size
        seL_chunk = left_se_B[i0:i0 + b]
        seR_chunk = right_se_B[i0:i0 + b]

        # Reset cached state between chunks so each goes through full init.
        dev.kpoint = None
        dev.newK_flag = None
        dev.newV_flag = None
        if hasattr(dev, "V"):
            del dev.V
        if hasattr(dev, "hd"):
            del dev.hd, dev.sd, dev.hl, dev.su, dev.sl, dev.hu

        # Runner restores lead.se to [n,n] after the last chunk, but mid-loop it
        # stacks; we replicate that here.
        if b > 1:
            dev.lead_L.se = seL_chunk
            dev.lead_R.se = seR_chunk
        else:
            dev.lead_L.se = seL_chunk[0]
            dev.lead_R.se = seR_chunk[0]

        dev.cal_green_function(
            energy=e_chunk,
            kpoint=[0.0, 0.0, 0.0],
            eta_device=0.0,
            block_tridiagonal=True,
            need_lesser=False,
            need_gr_lc=False,
        )
        tc_chunks.append(dev._cal_tc_().reshape(-1))
        dos_chunks.append(dev._cal_dos_().reshape(-1))
        ldos_one = dev._cal_ldos_()
        if ldos_one.ndim == 1:
            ldos_one = ldos_one.unsqueeze(0)
        ldos_chunks.append(ldos_one)

    tc_cat = torch.cat(tc_chunks)
    dos_cat = torch.cat(dos_chunks)
    ldos_cat = torch.cat(ldos_chunks)

    assert torch.allclose(tc_cat, tc_full, atol=1e-10, rtol=0)
    assert torch.allclose(dos_cat, dos_full, atol=1e-10, rtol=0)
    assert torch.allclose(ldos_cat, ldos_full, atol=1e-10, rtol=0)
