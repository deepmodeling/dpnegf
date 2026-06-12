"""Batched-energy contract for DeviceProperty.

The recursive Green's function kernel was widened to take a leading energy-batch
dim; this test pins that DeviceProperty.cal_green_function, _cal_tc_,
_cal_dos_, and _cal_ldos_ produce the same numbers whether the caller drives B
scalar energies in a loop or feeds a 1-D [B] energy tensor in one shot.

It also pins the runner-style chunking behavior (split [E] into chunks of size
e_batch_size, cat the per-chunk outputs back together).
"""

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

    def get_hs_device(self, kpoint, V, block_tridiagonal):
        return self._packed


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
    dev.device = "cpu"
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
