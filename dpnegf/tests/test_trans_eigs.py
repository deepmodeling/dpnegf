"""Independent analytic/dense references for transmission eigenchannels."""
from dataclasses import replace

import pytest
import torch

from dpnegf.negf.trans_eigs import (
    TransportContext, factor_broadening,
    transmission_eigenvalues, transmission_eigenchannels,
)

DT = torch.complex128


@pytest.mark.parametrize('energy,left,right,eta', [
    (0., 1., 1., 0.), (.7, .3, 1.2, 0.), (0., .3, 1.2, .001), (0., 0., 1., 0.),
])
def test_single_level(energy, left, right, eta):
    c = TransportContext(energy, [torch.zeros(1, 1, dtype=DT)],
                         [torch.ones(1, 1, dtype=DT)], [], [], [], [],
                         torch.tensor([[-.5j * left]], dtype=DT),
                         torch.tensor([[-.5j * right]], dtype=DT), eta_device=eta)
    result = transmission_eigenchannels(c)
    expected = left * right / (energy**2 + ((left + right) / 2 + eta)**2)
    torch.testing.assert_close(result['T_spectral'], torch.tensor(expected, dtype=torch.float64))
    assert result['normalization'] == 'spectral'
    if left:
        coeff = result['coefficients'][0]
        torch.testing.assert_close(coeff.abs().square().squeeze(),
                                   torch.tensor(left / (energy**2 + ((left + right) / 2 + eta)**2), dtype=torch.float64))
    else:
        assert result['tau_all'].shape == (0,)
        assert result['coefficients'][0].shape == (1, 0)


def _model(dtype=DT, device='cpu', rectangular=False, eta=.001):
    generator = torch.Generator().manual_seed(617)
    sizes = [3, 4, 2]
    def rand(*shape):
        return torch.randn(*shape, dtype=DT, generator=generator).to(dtype=dtype, device=device)
    hd = []
    for size in sizes:
        a = rand(size, size)
        hd.append((a + a.mH) * .2)
    hu = [rand(a, b) * .2 for a, b in zip(sizes[:-1], sizes[1:])]
    sd = [torch.eye(n, dtype=dtype, device=device) for n in sizes]
    su = [rand(a, b) * .02 for a, b in zip(sizes[:-1], sizes[1:])]
    # Non-contiguous left contact, unequal contact ranks, complex Gamma.
    bl = rand(2, 2) if not rectangular else rand(3, 3)
    br = rand(2, 2)
    left_idx = [2, 0] if not rectangular else [2, 0, 1]
    c = TransportContext(.17, hd, sd, hu, [x.mH for x in hu], su, [x.mH for x in su],
                         -.5j * (bl @ bl.mH), -.5j * (br @ br.mH),
                         left_indices=left_idx, right_indices=[1, 0], eta_device=eta,
                         orbital_map=[8, 0, 3, 7, 1, 6, 5, 2, 4])
    H = torch.block_diag(*hd)
    S = torch.block_diag(*sd)
    offsets = [0, 3, 7, 9]
    for i in range(2):
        rows, cols = slice(offsets[i], offsets[i + 1]), slice(offsets[i + 1], offsets[i + 2])
        H[rows, cols], H[cols, rows] = hu[i], hu[i].mH
        S[rows, cols], S[cols, rows] = su[i], su[i].mH
    li = c.left_indices
    ri = c.right_indices + 7
    SL = torch.zeros_like(H)
    SR = torch.zeros_like(H)
    SL[li[:, None], li] = c.left_se
    SR[ri[:, None], ri] = c.right_se
    M = c.z * S - H - SL - SR
    G = torch.linalg.solve(M, torch.eye(9, dtype=dtype, device=device))
    return c, H, S, SL, SR, G, li, ri


@pytest.mark.parametrize('incoming', ['left', 'right'])
@pytest.mark.parametrize('rectangular', [False, True])
def test_dense_complex_nonorthogonal_reference(incoming, rectangular):
    c, H, S, SL, SR, G, li, ri = _model(rectangular=rectangular)
    result = transmission_eigenchannels(c, incoming=incoming)
    coeff = torch.cat(result['coefficients'])
    gin, gout = (1j * (SL - SL.mH), 1j * (SR - SR.mH)) if incoming == 'left' else (1j * (SR - SR.mH), 1j * (SL - SL.mH))
    raw = torch.trace(gout @ G @ gin @ G.mH).real
    torch.testing.assert_close(result['T_spectral'], raw, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(coeff.mH @ gout @ coeff, torch.diag(result['tau_all']).to(DT))
    if result['input_rank'] <= result['output_rank']:
        torch.testing.assert_close(coeff @ coeff.mH, G @ gin @ G.mH)
    assert result['diagnostics']['solve_relative_residual'] < 1e-15
    values = transmission_eigenvalues(c, incoming=incoming)
    torch.testing.assert_close(values['tau_all'], result['tau_all'])
    assert 'coefficients' not in values
    assert result['n_implicit_zero_inputs'] == max(result['input_rank'] - result['output_rank'], 0)
    original = transmission_eigenchannels(c, incoming=incoming, n_channels=1, return_format='original')
    torch.testing.assert_close(original['coefficients'], coeff[c.orbital_map.argsort(), :1])
    selected = transmission_eigenchannels(c, incoming=incoming, channel_indices=[1])
    torch.testing.assert_close(torch.cat(selected['coefficients']), coeff[:, 1:2])
    # Independent Lowdin representation, with dense transformed contacts.
    eig, Q = torch.linalg.eigh(S)
    root = (Q * eig.sqrt()) @ Q.mH
    inverse = (Q * eig.rsqrt()) @ Q.mH
    ortho = TransportContext(c.energy, [inverse @ H @ inverse], [torch.eye(9, dtype=DT)],
                             [], [], [], [], inverse @ SL @ inverse, inverse @ SR @ inverse,
                             eta_device=c.eta_device)
    lowdin = transmission_eigenchannels(ortho, incoming=incoming)
    torch.testing.assert_close(lowdin['tau_all'][:len(result['tau_all'])], result['tau_all'])
    transformed = root @ coeff
    # Compare amplitude-weighted channel subspace, independent of phases/factor gauge.
    torch.testing.assert_close(transformed @ transformed.mH,
                               lowdin['coefficients'][0] @ lowdin['coefficients'][0].mH)


@pytest.mark.parametrize('incoming', ['left', 'right'])
def test_reuse_full_rgf_and_batch(tmp_path, incoming):
    from .test_device_property_batched import _make_deviceprop
    c, H, S, SL, SR, G, li, ri = _model()
    # Pad according to the existing boundary embedding, including zero rows.
    left = torch.zeros(3, 3, dtype=DT)
    left[c.left_indices[:, None], c.left_indices] = c.left_se
    right = torch.zeros(2, 2, dtype=DT)
    right[c.right_indices[:, None], c.right_indices] = c.right_se
    dev = _make_deviceprop([3, 4, 2], c.hd, c.sd, c.hl, c.hu, c.sl, c.su)
    dev.results_path = str(tmp_path)
    energies = torch.tensor([.17, .25, .31])
    dev.lead_L.se = left.expand(3, -1, -1)
    dev.lead_R.se = right.expand(3, -1, -1)
    dev.cal_green_function(energies, [0, 0, 0], eta_device=c.eta_device,
                           gf_cal_mode='transmission_only', need_trans_eigs=True)
    assert 'grd' in dev.greenfuncs  # Eigen outputs force full RGF.
    results = dev.cal_transmission_eigs(channels=True, n_channels=1, incoming=incoming)
    for energy, actual in zip(energies, results):
        reference = transmission_eigenchannels(replace(c, energy=energy), n_channels=1, incoming=incoming)
        torch.testing.assert_close(actual['tau_all'][:2], reference['tau_all'])
        # A full zero-padded contact has the same physical coefficient projectors.
        a, b = torch.cat(actual['coefficients']), torch.cat(reference['coefficients'])
        torch.testing.assert_close(a @ a.mH, b @ b.mH)
    dev.release_greenfuncs()
    with pytest.raises(RuntimeError):
        dev.cal_transmission_eigs()


def test_psd_thresholds_selection_and_empty_rank():
    with pytest.raises(ValueError, match='positive semidefinite'):
        factor_broadening(torch.diag(torch.tensor([1., -.01], dtype=DT)))
    with pytest.raises(ValueError, match='non-Hermitian'):
        factor_broadening(torch.tensor([[1., 1j], [1j, 1.]], dtype=DT))
    with pytest.raises(ValueError, match='Non-finite'):
        factor_broadening(torch.tensor([[float('nan')]], dtype=DT))
    factor, diagnostic = factor_broadening(torch.diag(torch.tensor([1., 1e-8, -1e-15], dtype=DT)), rank_rtol=1e-6)
    assert factor.shape == (3, 1)
    assert diagnostic['discarded_positive_count'] == 1
    c, *_ = _model(rectangular=True)
    result = transmission_eigenvalues(c, rank_rtol=.9)
    assert result['diagnostics']['transmission_absolute_error'] > 1e-6
    for options in ({'channel_indices': [0], 'n_channels': 1}, {'channel_indices': [-1]},
                    {'channel_indices': [.5]}, {'channel_indices': [0, 0]}, {'n_channels': -1}):
        with pytest.raises(ValueError):
            transmission_eigenchannels(c, **options)
    assert torch.cat(transmission_eigenchannels(c, n_channels=0)['coefficients']).shape == (9, 0)
    with pytest.raises(ValueError, match='cross-block'):
        replace(c, left_se=torch.eye(4, dtype=DT))
    with pytest.raises(ValueError, match='integer indices'):
        replace(c, left_indices=[0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match='integer indices'):
        replace(c, orbital_map=[x + .1 for x in range(9)])


def test_context_rejects_broadcast_blocks_and_invalid_cache():
    c, *_ = _model()
    for changes in ({'sd': [torch.ones(1, dtype=DT)] + c.sd[1:]},
                    {'hu': [torch.ones(1, 4, dtype=DT), c.hu[1]]},
                    {'gr_left': [torch.ones(1, dtype=DT)] * 3},
                    {'gr_left': [torch.full_like(h, float('nan')) for h in c.hd]},
                    {'g_trans': torch.ones(2, 3, dtype=DT)},
                    {'g_trans': torch.full((3, 2), float('nan'), dtype=DT)}):
        with pytest.raises(ValueError):
            replace(c, **changes)


@pytest.mark.parametrize('rectangular', [False, True])
def test_right_incidence_reuses_g_trans(monkeypatch, rectangular):
    from dpnegf.negf.trans_eigs import _ContactSolver
    c, _, _, _, _, G, _, _ = _model(rectangular=rectangular)
    cached = replace(c, g_trans=G[:3, 7:])
    expected = transmission_eigenchannels(c, incoming='right', n_channels=1)
    original_solve = _ContactSolver.solve
    calls = []
    def solve(self, source, *, blocks=False):
        calls.append((source.shape[1], blocks))
        return original_solve(self, source, blocks=blocks)
    monkeypatch.setattr(_ContactSolver, 'solve', solve)
    values = transmission_eigenvalues(cached, incoming='right')
    assert calls == []  # No new block solve is needed for the spectrum.
    actual = transmission_eigenchannels(cached, incoming='right', n_channels=1)
    assert calls == [(1, True)]  # Only the selected channel source is solved.
    torch.testing.assert_close(values['tau_all'], expected['tau_all'])
    torch.testing.assert_close(actual['T_caroli_raw'], expected['T_caroli_raw'])
    a, b = torch.cat(actual['coefficients']), torch.cat(expected['coefficients'])
    torch.testing.assert_close(a @ a.mH, b @ b.mH)
    assert actual['diagnostics']['solve_relative_residual'] < 1e-15


@pytest.mark.parametrize('incoming', ['left', 'right'])
@pytest.mark.parametrize('rectangular', [False, True])
@pytest.mark.parametrize('cache_g_trans', [False, True])
def test_channels_reuse_gr_left_without_factorization(monkeypatch, incoming, rectangular, cache_g_trans):
    from dpnegf.negf.recursive_green_cal import recursive_gf
    from dpnegf.negf.trans_eigs import _ContactSolver
    c, _, _, SL, SR, G, li, ri = _model(rectangular=rectangular)
    full = recursive_gf(c.energy, c.hl, c.hd, c.hu, c.sd, c.su, c.sl,
                        SL[:3, :3], SR[7:, 7:], eta=c.eta_device, keep_gr_left=True)
    cached = replace(c, gr_left=full[5], g_trans=full[0] if cache_g_trans else None)
    expected = transmission_eigenchannels(c, incoming=incoming, n_channels=2)

    def forbidden(*args, **kwargs):
        raise AssertionError('Cached gr_left must avoid new matrix factorization or solves')
    for name in ('lu_factor', 'lu_solve', 'solve', 'inv'):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    actual = transmission_eigenchannels(cached, incoming=incoming, n_channels=2)
    torch.testing.assert_close(actual['tau_all'], expected['tau_all'])
    a, b = torch.cat(actual['coefficients']), torch.cat(expected['coefficients'])
    torch.testing.assert_close(a @ a.mH, b @ b.mH)
    assert actual['diagnostics']['solve_relative_residual'] < 1e-15

    # Check arbitrary complex multiple RHS directly against the dense Green,
    # including the right-injection boundary and non-contiguous contact indices.
    solver = _ContactSolver(cached, incoming, retain=True)
    assert solver.order == [0, 1, 2]
    indices, out_indices = (li, ri) if incoming == 'left' else (ri, li)
    rhs = torch.arange(len(indices) * 3, dtype=torch.float64).reshape(-1, 3).to(DT) * (1 + .3j)
    dense = G[:, indices] @ rhs
    torch.testing.assert_close(torch.cat(solver.solve(rhs, blocks=True)), dense)
    torch.testing.assert_close(solver.solve(rhs), dense[out_indices])
    assert solver.factors == [] and solver.transfers == []


def test_device_eigenchannels_keep_green_solve_state(tmp_path):
    from .test_device_property_batched import _make_deviceprop
    c, *_ = _model()
    dev = _make_deviceprop([3, 4, 2], c.hd, c.sd, c.hl, c.hu, c.sl, c.su)
    dev.results_path = str(tmp_path)
    dev.lead_L.se = -.5j * torch.eye(3, dtype=DT)
    dev.lead_R.se = -.5j * torch.eye(2, dtype=DT)
    energies = torch.tensor([.17, .25], dtype=torch.float64)
    dev.lead_L.se = dev.lead_L.se.expand(2, -1, -1).clone()
    dev.lead_R.se = dev.lead_R.se.expand(2, -1, -1).clone()
    dev.cal_green_function(energies, [0, 0, 0], need_trans_eigs=True)
    before = dev.cal_transmission_eigs(channels=True, n_channels=1)
    dev.lead_L.se.mul_(2)
    dev.lead_R.se = -.7j * torch.eye(2, dtype=DT)
    dev.E_ref = 10.
    energies.add_(1)
    after = dev.cal_transmission_eigs(channels=True, n_channels=1)
    for a, b in zip(before, after):
        torch.testing.assert_close(a['tau_all'], b['tau_all'])
        torch.testing.assert_close(torch.cat(a['coefficients']), torch.cat(b['coefficients']))
        torch.testing.assert_close(a['energy'], b['energy'])
        assert a['E_ref'] == b['E_ref']
    with pytest.raises(AssertionError):
        dev.cal_green_function(0., [0, 0], need_trans_eigs=True)
    with pytest.raises(RuntimeError):
        dev.cal_transmission_eigs()


def test_transmission_overflow_is_explicit():
    c = TransportContext(0., [torch.zeros(1, 1, dtype=DT)],
                         [torch.ones(1, 1, dtype=DT)], [], [], [], [],
                         torch.tensor([[-1e20j]], dtype=DT),
                         torch.tensor([[-1e20j]], dtype=DT),
                         gr_left=[torch.tensor([[1e300]], dtype=DT)])
    with pytest.raises(ValueError, match='Non-finite'):
        transmission_eigenvalues(c)


def test_conservation_with_loop_phase_and_finite_absorption():
    # Intra-block triangle has a gauge-invariant magnetic flux.
    c, *_ = _model(eta=0.)
    c.hd[1][0, 1] += .2j
    c.hd[1][1, 0] -= .2j
    left = transmission_eigenvalues(c, incoming='left')
    right = transmission_eigenvalues(c, incoming='right')
    torch.testing.assert_close(left['tau_all'], right['tau_all'])
    absorb = replace(c, eta_device=.1)
    left = transmission_eigenvalues(absorb, incoming='left')
    right = transmission_eigenvalues(absorb, incoming='right')
    assert abs(left['T_spectral'] - right['T_spectral']) > 1e-5


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_backend_precision(dtype, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    context, *_ = _model(dtype=dtype, device=device)
    result = transmission_eigenchannels(context, n_channels=1)
    reference = transmission_eigenchannels(_model()[0], n_channels=1)
    torch.testing.assert_close(result['tau_all'].cpu().double(), reference['tau_all'], atol=3e-6, rtol=3e-5)
    assert result['coefficients'][0].device.type == device
    assert result['diagnostics']['solve_relative_residual'] < (1e-7 if dtype == torch.complex64 else 1e-15)


@pytest.mark.parametrize('outputs,btd,chunk', [
    ({'trans_eigvals': True}, False, 2),
    ({'trans_eigchannel': True}, True, 1),
    ({'trans_eigvals': True, 'trans_eigchannel': True, 'tc': True}, True, 2),
])
def test_input_json_runner(tmp_path, outputs, btd, chunk):
    import json
    from pathlib import Path
    from dpnegf.entrypoints.run import run
    root = Path(__file__).resolve().parent / 'data/test_negf/test_negf_run'
    config = json.loads((root / 'negf_chain_new.json').read_text())
    config['structure'] = str(root / 'chain.vasp')
    task = config['task_options']
    structure_path = str(root / 'chain.vasp')
    if btd and not outputs.get('tc'):
        from ase.io import read, write
        atoms = read(structure_path)
        atoms = atoms[[0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]]
        structure_path = str(tmp_path / 'unsorted.xyz')
        write(structure_path, atoms)
    config['structure'] = structure_path
    if outputs.get('tc'):
        task['btd_initialization'] = 'direct'
    for key in list(task):
        if key.startswith('out_') or key in ('emin', 'emax', 'espacing', 'sgf_solver'):
            task.pop(key)
    task.update(output_options=outputs, block_tridiagonal=btd,
                energy_grid={'method': 'uniform', 'emin': -.05, 'emax': .05, 'num_points': 3},
                trans_eigs_options={'n_channels': 1},
                rgf_options={'device': 'cpu', 'e_batch_size': chunk},
                self_energy_options={'numba_jit': False,
                                     'parallel': {'cpu_budget': 1, 'n_workers': 1, 'blas_threads': 1}})
    if not outputs.get('trans_eigchannel'):
        # Channel selection is ignored when only eigenvalues are requested.
        task['trans_eigs_options']['channel_indices'] = [0]
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(config))
    run(INPUT=str(path), structure=structure_path, init_model=str(root / 'nnsk_C_new.json'),
        output=str(tmp_path / 'out'), log_level=20, log_path=str(tmp_path / 'run.log'))
    result = torch.load(tmp_path / 'out/results/negf.out.pth', weights_only=False)
    k = str(result['k'][0])
    for flag in ('trans_eigvals', 'trans_eigchannel'):
        assert (flag in result) == outputs.get(flag, False)
        if flag not in result:
            continue
        records = result[flag][k]
        assert len(records) == 3
        for entry in records:
            record = entry
            if flag == 'trans_eigchannel':
                assert set(entry) == {'energy', 'kpoint', 'path'}
                record = torch.load(tmp_path / 'out/results' / entry['path'], weights_only=False)
                torch.testing.assert_close(record['energy'], entry['energy'])
            assert abs(record['tau_all'][0] - 1) < 2e-5
            assert record['incoming'] == 'left'
            assert record['orbital_mapping']['status'] == 'known'
            assert record['orbital_mapping_status'] == 'known'
            assert record['orbital_mapping']['basis']['convention'] == 'dptb-OrbitalMapper-real-spherical-v1'
            torch.testing.assert_close(record['inverse_orbital_map'], record['orbital_map'].argsort())
            if flag == 'trans_eigchannel':
                assert record['coefficients'].shape[1] == 1
                assert record['basis_order'] == 'original'
                if not outputs.get('tc'):
                    assert record['orbital_map'].tolist() == [3, 2, 1, 0]
                assert record['diagnostics']['solve_relative_residual'] < 1e-12
    if outputs.get('tc'):
        torch.testing.assert_close(torch.stack([r['T_spectral'] for r in result['trans_eigvals'][k]]), result['T_k'][k])
    assert 'transmission-only recursive' not in (tmp_path / 'run.log').read_text()


def test_runner_streams_saves_and_releases_each_energy(tmp_path, monkeypatch):
    import weakref
    from dpnegf.negf.device_property import DeviceProperty
    from dpnegf.runner.NEGF import NEGF
    raw_refs, saved_refs = [], []
    def make_record(i):
        coefficients = torch.ones(4, 1, dtype=DT) * (i + 1)
        raw_refs.append(weakref.ref(coefficients))
        return dict(energy=torch.tensor(float(i)), kpoint=torch.zeros(3),
                    coefficients=coefficients, tau_all=torch.ones(1),
                    selected_indices=torch.tensor([0]), tau_selected=torch.ones(1))
    def iterator(**options):
        for i in range(3):
            # The previous energy must already be on disk with both copies released.
            assert len(list(tmp_path.rglob('*.pth'))) == i
            assert all(ref() is None for ref in raw_refs + saved_refs)
            yield make_record(i)
    original_save = torch.save
    def save(record, path):
        assert record['coefficients'].device.type == 'cpu'
        assert record['orbital_atoms'] is None
        assert record['block_atom_map'] is None
        saved_refs.append(weakref.ref(record['coefficients']))
        original_save(record, path)
    monkeypatch.setattr(torch, 'save', save)
    runner = NEGF.__new__(NEGF)
    runner.trans_eigs_options = {'n_channels': 1}
    runner.out_trans_eigvals = runner.out_trans_eigchannel = True
    runner.eta_lead = 1e-5
    runner.results_path = str(tmp_path)
    runner.out = {}
    runner.deviceprop = DeviceProperty.__new__(DeviceProperty)
    runner.deviceprop.results_path = str(tmp_path)
    runner.deviceprop.iter_transmission_eigs = iterator
    runner.store_trans_eigs('k', 0)
    assert all(ref() is None for ref in raw_refs + saved_refs)
    assert len(runner.out['trans_eigchannel']['k']) == 3
    assert all('coefficients' not in r for r in runner.out['trans_eigvals']['k'])


def test_two_matched_chains_and_known_scatterer():
    # For hopping -1 at E=0 a single onsite U has T=4/(4+U^2).
    eye = torch.eye(2, dtype=DT)
    diagonal = [torch.zeros(2, 2, dtype=DT) for _ in range(5)]
    c = TransportContext(0., diagonal, [eye] * 5, [-eye] * 4, [-eye] * 4,
                         [eye * 0] * 4, [eye * 0] * 4, -1j * eye, -1j * eye)
    torch.testing.assert_close(transmission_eigenvalues(c)['tau_all'], torch.ones(2, dtype=torch.float64))
    diagonal[2][1, 1] = 2.
    result = transmission_eigenchannels(c)
    torch.testing.assert_close(result['tau_all'], torch.tensor([1., .5], dtype=torch.float64))
    coeff = torch.cat(result['coefficients'])
    source_gamma = torch.zeros(10, 10, dtype=DT)
    source_gamma[:2, :2] = 2 * eye
    M = torch.block_diag(*diagonal).neg()
    for i in range(4):
        M[2*i:2*i+2, 2*i+2:2*i+4] = eye
        M[2*i+2:2*i+4, 2*i:2*i+2] = eye
    M[:2, :2] += 1j * eye
    M[-2:, -2:] += 1j * eye
    G = torch.linalg.solve(M, torch.eye(10, dtype=DT))
    torch.testing.assert_close(coeff @ coeff.mH, G @ source_gamma @ G.mH)


@pytest.mark.parametrize('nblocks,energy', [(5, .2), (10, 0.)])
def test_cnt55_periodic_pi_model(nblocks, energy):
    from examples.transmission_eigenchannels.run_cnt55 import cnt_context
    c, atoms = cnt_context(nblocks=nblocks, energy=energy)
    result = transmission_eigenchannels(c, n_channels=2, return_format='original')
    torch.testing.assert_close(result['tau_all'][:2], torch.ones(2, dtype=torch.float64), atol=2e-6, rtol=0)
    assert abs(result['T_spectral'] - 2) < 4e-6
    assert result['coefficients'].shape == (len(atoms), 2)


def test_no_dense_device_allocation_for_block_transport():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    c, *_ = _model()
    class NoDeviceSquare(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if torch.is_tensor(value):
                    assert value.shape[-2:] != (9, 9), f'Dense device allocation in {func}'
            return result
    with NoDeviceSquare():
        transmission_eigenvalues(c)
        result = transmission_eigenchannels(c, n_channels=1)
    assert torch.cat(result['coefficients']).shape == (9, 1)


@pytest.mark.parametrize('kwargs', [{}, {'orbital_map': None}])
def test_standalone_context_missing_mapping_is_unknown(kwargs):
    c = TransportContext(0., [torch.zeros(1, 1, dtype=DT)],
                         [torch.ones(1, 1, dtype=DT)], [], [], [], [],
                         torch.tensor([[-1j]], dtype=DT),
                         torch.tensor([[-1j]], dtype=DT), **kwargs)
    assert c.orbital_map is None
    assert c.orbital_map_known is False
    values = transmission_eigenvalues(c)
    assert values['orbital_mapping_status'] == 'unknown'
    assert values['orbital_map'] is values['inverse_orbital_map'] is None
    channels = transmission_eigenchannels(c, return_format='blocks')
    assert channels['basis_order'] == 'blocks'
    with pytest.raises(ValueError, match='unknown'):
        transmission_eigenchannels(c, return_format='original')
    with pytest.raises(ValueError, match='explicit orbital_map'):
        replace(c, orbital_map_known=True)
    known = replace(c, orbital_map=[0], orbital_map_known=None)
    assert known.orbital_map_known is True
    original = transmission_eigenchannels(known, return_format='original')
    assert original['orbital_mapping_status'] == 'known'
    torch.testing.assert_close(original['coefficients'], channels['coefficients'][0])
    with pytest.raises(ValueError, match='cannot specify'):
        replace(known, orbital_map_known=False)
