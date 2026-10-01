import pytest
import torch

from dpnegf.negf.recursive_green_cal import (
    recursive_gf,
    recursive_gf_transmission_only,
)

from .test_recursive_gf_batched import _make_btd_inputs


def _assemble_full_matrix(diagonal, lower, upper):
    offsets = [0]
    for block in diagonal:
        offsets.append(offsets[-1] + block.shape[-1])
    matrix = torch.zeros(
        offsets[-1],
        offsets[-1],
        dtype=diagonal[0].dtype,
        device=diagonal[0].device,
    )
    for block_index, block in enumerate(diagonal):
        start, end = offsets[block_index:block_index + 2]
        matrix[start:end, start:end] = block
        if block_index < len(upper):
            next_end = offsets[block_index + 2]
            matrix[start:end, end:next_end] = upper[block_index]
            matrix[end:next_end, start:end] = lower[block_index]
    return matrix


@pytest.mark.parametrize("block_sizes", [[8, 6, 7, 8], [8, 8, 8, 8]])
@pytest.mark.parametrize("batched", [False, True])
def test_transmission_only_matches_general_rgf(block_sizes, batched):
    batch_size = 5
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        batch_size, block_sizes, seed=21
    )

    generator = torch.Generator(device="cpu").manual_seed(22)
    for block_index in range(len(su)):
        overlap = (
            torch.randn(
                *su[block_index].shape,
                generator=generator,
                dtype=torch.float64,
            )
            + 1j
            * torch.randn(
                *su[block_index].shape,
                generator=generator,
                dtype=torch.float64,
            )
        ).to(torch.complex128)
        su[block_index] = 0.01 * overlap
        sl[block_index] = su[block_index].mH

    if batched:
        energy = energies
    else:
        energy = energies[0]
        left_se = left_se[0]
        right_se = right_se[0]

    reference = recursive_gf(
        energy=energy,
        hl=hl,
        hd=hd,
        hu=hu,
        sd=sd,
        su=su,
        sl=sl,
        left_se=left_se,
        right_se=right_se,
        eta=1e-5,
        need_lesser=False,
        need_greater=False,
        need_gr_lc=False,
        keep_gr_left=False,
    )[0]
    actual = recursive_gf_transmission_only(
        energy=energy,
        hl=hl,
        hd=hd,
        hu=hu,
        sd=sd,
        su=su,
        sl=sl,
        left_se=left_se,
        right_se=right_se,
        eta=1e-5,
    )

    assert actual.shape == reference.shape
    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_transmission_only_matches_direct_inverse():
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        1, [4, 3, 5], seed=25
    )
    energy = energies[0]
    left_se = left_se[0]
    right_se = right_se[0]

    full_h = _assemble_full_matrix(hd, hl, hu)
    full_s = _assemble_full_matrix(sd, sl, su)
    effective = (energy + 1j * 1e-5) * full_s - full_h
    effective[:4, :4] -= left_se
    effective[-5:, -5:] -= right_se
    reference = torch.linalg.inv(effective)[:4, -5:]

    actual = recursive_gf_transmission_only(
        energy=energy,
        hl=hl,
        hd=hd,
        hu=hu,
        sd=sd,
        su=su,
        sl=sl,
        left_se=left_se,
        right_se=right_se,
        eta=1e-5,
    )

    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_transmission_only_single_block_matches_general_rgf():
    batch_size = 4
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        batch_size, [8], seed=23
    )

    reference = recursive_gf(
        energy=energies,
        hl=hl,
        hd=hd,
        hu=hu,
        sd=sd,
        su=su,
        sl=sl,
        left_se=left_se,
        right_se=right_se,
        eta=1e-5,
        keep_gr_left=False,
    )[0]
    actual = recursive_gf_transmission_only(
        energy=energies,
        hl=hl,
        hd=hd,
        hu=hu,
        sd=sd,
        su=su,
        sl=sl,
        left_se=left_se,
        right_se=right_se,
        eta=1e-5,
    )

    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transmission_only_cuda_matches_cpu():
    batch_size = 8
    block_sizes = [12, 10, 8, 12]
    cpu_inputs = _make_btd_inputs(
        batch_size, block_sizes, seed=24, device="cpu"
    )
    cuda_inputs = _make_btd_inputs(
        batch_size, block_sizes, seed=24, device="cuda"
    )

    cpu_result = recursive_gf_transmission_only(
        energy=cpu_inputs[-1],
        hl=cpu_inputs[2],
        hd=cpu_inputs[0],
        hu=cpu_inputs[3],
        sd=cpu_inputs[1],
        su=cpu_inputs[5],
        sl=cpu_inputs[4],
        left_se=cpu_inputs[6],
        right_se=cpu_inputs[7],
    )
    cuda_result = recursive_gf_transmission_only(
        energy=cuda_inputs[-1],
        hl=cuda_inputs[2],
        hd=cuda_inputs[0],
        hu=cuda_inputs[3],
        sd=cuda_inputs[1],
        su=cuda_inputs[5],
        sl=cuda_inputs[4],
        left_se=cuda_inputs[6],
        right_se=cuda_inputs[7],
    )

    assert torch.allclose(cuda_result.cpu(), cpu_result, atol=1e-8, rtol=1e-8)