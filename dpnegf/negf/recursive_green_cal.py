import torch.linalg as tLA
import torch

def recursive_gf_cal(energy, mat_l_list, mat_d_list, mat_u_list,
                     sd, su, sl, s_in=0, s_out=0, eta=1e-5,
                     need_lesser=False, need_greater=False, need_gr_lc=False,
                     stacked=False, keep_gr_left=True):
    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    Batched form: every tensor carries a leading energy-batch dim B. Diagonal
    blocks have shape ``[B, n_q, n_q]`` and the energy argument is ``[B]``.
    Scalar callers go through :func:`recursive_gf`, which broadcasts to B=1
    and squeezes the leading dim back out.

    In order to get the electron correlation function output, the parameters s_in has to be set.
    For the hole correlation function, the parameter s_out has to be set.
    By default, the function would return the retarded Green's function blocks.

    Parameters
    ----------
    energy : torch.Tensor (dtype=torch.complex)
        Energy array of shape ``[B]``.
    mat_d_list : list of torch.Tensor (dtype=torch.complex)
        List of diagonal blocks, each of shape ``[B, n_q, n_q]``. When
        ``stacked=True``, a single ``[K, B, n, n]`` tensor with K = num blocks.
    mat_u_list : list of torch.Tensor (dtype=torch.complex)
        List of upper-diagonal blocks, each of shape ``[B, n_q, n_{q+1}]``.
        When ``stacked=True``, a single ``[K-1, B, n, n]`` tensor.
    mat_l_list : list of torch.Tensor (dtype=torch.complex)
        List of lower-diagonal blocks, each of shape ``[B, n_{q+1}, n_q]``.
        When ``stacked=True``, a single ``[K-1, B, n, n]`` tensor.
    s_in :
         (Default value = 0). When ``need_lesser`` is True, a list of
         ``[B, n_q, n_q]`` tensors.
    s_out :
         (Default value = 0). When ``need_greater`` is True, a list of
         ``[B, n_q, n_q]`` tensors.
    eta :
         (Default value = 0.000001j)
    need_lesser : bool, optional
        Whether to calculate the lesser Green's function, by default False.
        Lesser Green's function is used for electron density and current density calculation.
    need_greater : bool, optional
        Whether to calculate the greater Green's function, by default False.
        Greater Green's function is used for hole density and phase-breaking scattering case.
    need_gr_lc : bool, optional
        Whether to calculate the last column blocks of the retarded Green's function responsible for transmission, by default True
        gr_lc is used for lead spectral function A_L/ A_R = G^r * Gamma_L/R * G^a calculation.
        Although set need_gr_lc to True would not increase the computational cost of the recursive Green's function algorithm, it would increase the memory cost.
        If the memory cost is a concern, it is recommended to set need_gr_lc to False.
    stacked : bool, optional
        When True, ``mat_d_list``/``mat_u_list``/``mat_l_list`` and
        ``sd``/``su``/``sl`` are single 4-D tensors of shape ``[K,B,n,n]``
        (resp. ``[K-1,B,n,n]``) instead of Python lists. The wrapper sets
        this automatically when all device blocks share an `n`; the block
        outputs are unbound back to lists so the caller contract is unchanged.

    Returns
    -------
    All returned blocks carry a leading batch dim ``B``.
    """
    if need_lesser:
        assert isinstance(s_in, list), "Lesser Green's function calculation requires s_in to be a list of coupling matrices"
    if need_greater:
        assert isinstance(s_out, list), "Greater Green's function calculation requires s_out to be a list of coupling matrices"

    # ------------------------------------------------------------------
    # Uniform-block path
    # ------------------------------------------------------------------
    if stacked:
        # mat_*/s* are 4-D tensors; do the energy shift in three fused ops
        # over the leading-K dim instead of K Python iterations.
        e_bcast = (energy + 1j * eta).view(1, -1, 1, 1)
        mat_d = mat_d_list - e_bcast * sd
        mat_l = mat_l_list - e_bcast * sl
        mat_u = mat_u_list - e_bcast * su

        num_of_matrices = mat_d.shape[0]
        ref = mat_d
        B = ref.shape[1]
        n = ref.shape[-1]
        # Single identity reused across all K forward solves.
        eye_bnn = torch.eye(n, dtype=ref.dtype, device=ref.device).expand(B, n, n)

        # ------------- left-connected retarded Green's function ------------
        gr_left = [None] * num_of_matrices
        gr_left[0] = tLA.solve(-mat_d[0], eye_bnn)
        for q in range(num_of_matrices - 1):  # (B2)
            gr_left[q + 1] = tLA.solve(
                -mat_d[q + 1] - mat_l[q] @ gr_left[q] @ mat_u[q],
                eye_bnn,
            )
        # mat_d is dead after the forward sweep — backward sweep only reads mat_l/mat_u.
        del mat_d

        grl = [None] * (num_of_matrices - 1)
        gru = [None] * (num_of_matrices - 1)
        grd = [None] * num_of_matrices
        grd[-1] = gr_left[-1].clone()
        g_trans = gr_left[-1].clone()
        gr_lc = [g_trans] if need_gr_lc else None
        for q in range(num_of_matrices - 2, -1, -1):
            gU = gr_left[q] @ mat_u[q]            # hoisted: used 2-3x below
            grl[q] = grd[q + 1] @ mat_l[q] @ gr_left[q]  # (B5)
            gru[q] = gU @ grd[q + 1]                     # (B6)
            grd[q] = gr_left[q] + gU @ grl[q]            # (B4)
            g_trans = gU @ g_trans
            if need_gr_lc:
                gr_lc.append(g_trans)
            del gU
        if need_gr_lc:
            gr_lc.reverse()
        if not need_lesser and not need_greater:
            # Stacked path: mat_l/mat_u are 4-D and can't be slice-freed mid-loop;
            # they are dead now if no lesser/greater pass will read them.
            del mat_l, mat_u

        gnd = gnl = gnu = gin_left = None
        if need_lesser:
            gin_left = [None] * num_of_matrices
            # G^< = G^r * Sigma^< * G^a
            gin_left[0] = gr_left[0] @ s_in[0] @ gr_left[0].mH
            for q in range(num_of_matrices - 1):
                sla2 = mat_l[q] @ gin_left[q] @ mat_u[q]
                gin_left[q + 1] = gr_left[q + 1] @ (s_in[q + 1] + sla2) @ gr_left[q + 1].mH

            gnl = [None] * (num_of_matrices - 1)
            gnu = [None] * (num_of_matrices - 1)
            gnd = [None] * num_of_matrices
            gnd[-1] = gin_left[-1].clone()
            for q in range(num_of_matrices - 2, -1, -1):
                gLmH = mat_l[q] @ gr_left[q].mH         # hoisted: used twice
                gnl[q] = grd[q + 1] @ mat_l[q] @ gin_left[q] + gnd[q + 1] @ gLmH  # (B10)
                gnd[q] = gin_left[q] + \
                         gr_left[q] @ mat_u[q] @ gnd[q + 1] @ gLmH + \
                         (gin_left[q] @ mat_u[q] @ gru[q].mH) + \
                         (gru[q] @ mat_l[q] @ gin_left[q])  # (B11)
                gnu[q] = gnl[q].mH

        gpd = gpl = gpu = gip_left = None
        if need_greater:
            gip_left = [None] * num_of_matrices
            gip_left[0] = gr_left[0] @ s_out[0] @ gr_left[0].conj()
            for q in range(num_of_matrices - 1):
                sla2 = mat_l[q] @ gip_left[q] @ mat_u[q].conj()
                gip_left[q + 1] = gr_left[q + 1] @ (s_out[q + 1] + sla2) @ gr_left[q + 1].conj()

            gpl = [None] * (num_of_matrices - 1)
            gpu = [None] * (num_of_matrices - 1)
            gpd = [None] * num_of_matrices
            gpd[-1] = gip_left[-1].clone()
            for q in range(num_of_matrices - 2, -1, -1):
                lcgc = mat_l[q].conj() @ gr_left[q].conj()  # hoisted: used twice
                gpl[q] = grd[q + 1] @ mat_l[q] @ gip_left[q] + gpd[q + 1] @ lcgc
                gpd[q] = gip_left[q] + \
                         gr_left[q] @ mat_u[q] @ gpd[q + 1] @ lcgc + \
                         (gip_left[q] @ mat_u[q].conj() @ grl[q].conj()) + \
                         (gru[q] @ mat_l[q] @ gip_left[q])
                gpu[q] = gpl[q].mH

        if not keep_gr_left:
            gr_left = None
        return _pack_ans(g_trans, gr_lc, grd, grl, gru, gr_left,
                         gnd, gnl, gnu, gin_left,
                         gpd, gpl, gpu, gip_left,
                         need_lesser, need_greater)

    # ------------------------------------------------------------------
    # Non-uniform-block path
    # ------------------------------------------------------------------
    e_bcast = (energy + 1j * eta).view(-1, 1, 1)

    for jj in range(len(mat_d_list)):
        # In-place: mat_d_list is a fresh tensor (wrapper's `* 1.` copy on D),
        # so we can fuse the energy shift without the e_bcast*sd transient.
        mat_d_list[jj].addcmul_(sd[jj], e_bcast, value=-1)
    for jj in range(len(mat_l_list)):
        mat_l_list[jj] = mat_l_list[jj] - e_bcast * sl[jj]
    for jj in range(len(mat_u_list)):
        mat_u_list[jj] = mat_u_list[jj] - e_bcast * su[jj]

    num_of_matrices = len(mat_d_list)
    mat_shapes = [item.shape for item in mat_d_list]  # [B, n_q, n_q]
    ref = mat_d_list[0]
    B = ref.shape[0]

    eye_cache = {}
    def _batched_eye(n):
        e = eye_cache.get(n)
        if e is None:
            e = torch.eye(n, dtype=ref.dtype, device=ref.device).expand(B, n, n)
            eye_cache[n] = e
        return e

    # ------------------ retarded Green's function ----------------------
    gr_left = [None] * num_of_matrices
    gr_left[0] = tLA.solve(-mat_d_list[0], _batched_eye(mat_shapes[0][-1]))
    mat_d_list[0] = None  # consumed; free immediately

    for q in range(num_of_matrices - 1):  # (B2)
        gr_left[q + 1] = tLA.solve(
            -mat_d_list[q + 1] - mat_l_list[q] @ gr_left[q] @ mat_u_list[q],
            _batched_eye(mat_shapes[q + 1][-1]),
        )
        mat_d_list[q + 1] = None  # consumed; backward sweep only reads mat_l/mat_u.

    grl = [None] * (num_of_matrices - 1)
    gru = [None] * (num_of_matrices - 1)
    grd = [None] * num_of_matrices
    grd[-1] = gr_left[-1].clone()
    g_trans = gr_left[-1].clone()
    gr_lc = [g_trans] if need_gr_lc else None
    # Slots that go dead at the end of iteration q:
    #   - mat_l_list[q], mat_u_list[q]: only re-read by the lesser/greater branches.
    #   - gr_left[q]: dead unless the lesser/greater branch will consume it OR
    #                 the caller asked us to keep the list intact.
    # Nulling per slot lets the caching allocator coalesce its free list inside the
    # loop instead of holding a long fragmented tail until the sweep ends.
    drop_lu = not need_lesser and not need_greater
    drop_gl = drop_lu and not keep_gr_left
    for q in range(num_of_matrices - 2, -1, -1):
        gU = gr_left[q] @ mat_u_list[q]                            # hoisted
        grl[q] = grd[q + 1] @ mat_l_list[q] @ gr_left[q]           # (B5)
        gru[q] = gU @ grd[q + 1]                                   # (B6)
        grd[q] = gr_left[q] + gU @ grl[q]                          # (B4)
        g_trans = gU @ g_trans
        if need_gr_lc:
            gr_lc.append(g_trans)
        del gU
        if drop_lu:
            mat_l_list[q] = None
            mat_u_list[q] = None
        if drop_gl:
            gr_left[q] = None
    if need_gr_lc:
        gr_lc.reverse()

    gnd = gnl = gnu = gin_left = None
    if need_lesser:
        assert isinstance(s_in, list), "need_lesser=True requires s_in to be a list of coupling matrices"
        gin_left = [None] * num_of_matrices
        # G^< = G^r * Sigma^< * G^a
        gin_left[0] = gr_left[0] @ s_in[0] @ gr_left[0].mH

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q] @ gin_left[q] @ mat_u_list[q]
            gin_left[q + 1] = gr_left[q + 1] @ (s_in[q + 1] + sla2) @ gr_left[q + 1].mH

        gnl = [None] * (num_of_matrices - 1)
        gnu = [None] * (num_of_matrices - 1)
        gnd = [None] * num_of_matrices
        gnd[-1] = gin_left[-1].clone()

        for q in range(num_of_matrices - 2, -1, -1):
            gLmH = mat_l_list[q] @ gr_left[q].mH                   # hoisted
            gnl[q] = grd[q + 1] @ mat_l_list[q] @ gin_left[q] + \
                     gnd[q + 1] @ gLmH                             # (B10)
            gnd[q] = gin_left[q] + \
                     gr_left[q] @ mat_u_list[q] @ gnd[q + 1] @ gLmH + \
                     (gin_left[q] @ mat_u_list[q] @ gru[q].mH) + \
                     (gru[q] @ mat_l_list[q] @ gin_left[q])        # (B11)
            gnu[q] = gnl[q].mH

    gpd = gpl = gpu = gip_left = None
    if need_greater:
        assert isinstance(s_out, list), "need_greater=True requires s_out to be a list of coupling matrices"
        gip_left = [None] * num_of_matrices
        gip_left[0] = gr_left[0] @ s_out[0] @ gr_left[0].conj()

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q] @ gip_left[q] @ mat_u_list[q].conj()
            gip_left[q + 1] = gr_left[q + 1] @ (s_out[q + 1] + sla2) @ gr_left[q + 1].conj()

        gpl = [None] * (num_of_matrices - 1)
        gpu = [None] * (num_of_matrices - 1)
        gpd = [None] * num_of_matrices
        gpd[-1] = gip_left[-1].clone()

        for q in range(num_of_matrices - 2, -1, -1):
            lcgc = mat_l_list[q].conj() @ gr_left[q].conj()        # hoisted
            gpl[q] = grd[q + 1] @ mat_l_list[q] @ gip_left[q] + \
                     gpd[q + 1] @ lcgc
            gpd[q] = gip_left[q] + \
                     gr_left[q] @ mat_u_list[q] @ gpd[q + 1] @ lcgc + \
                     (gip_left[q] @ mat_u_list[q].conj() @ grl[q].conj()) + \
                     (gru[q] @ mat_l_list[q] @ gip_left[q])
            gpu[q] = gpl[q].mH

    if not keep_gr_left:
        gr_left = None
    return _pack_ans(g_trans, gr_lc, grd, grl, gru, gr_left,
                     gnd, gnl, gnu, gin_left,
                     gpd, gpl, gpu, gip_left,
                     need_lesser, need_greater)


def _pack_ans(g_trans, gr_lc, grd, grl, gru, gr_left,
              gnd, gnl, gnu, gin_left,
              gpd, gpl, gpu, gip_left,
              need_lesser, need_greater):
    if not need_lesser and not need_greater:
        return g_trans, gr_lc, \
               grd, grl, gru, gr_left, \
               None, None, None, None, \
               None, None, None, None
    if need_lesser and not need_greater:
        return g_trans, gr_lc, \
               grd, grl, gru, gr_left, \
               gnd, gnl, gnu, gin_left, \
               None, None, None, None
    if not need_lesser and need_greater:
        return g_trans, gr_lc, \
               grd, grl, gru, gr_left, \
               None, None, None, None, \
               gpd, gpl, gpu, gip_left
    return g_trans, gr_lc, \
           grd, grl, gru, gr_left, \
           gnd, gnl, gnu, gin_left, \
           gpd, gpl, gpu, gip_left


def recursive_gf(energy, hl, hd, hu, sd, su, sl, left_se, right_se, seP=None, E_ref=0.0, s_in=0, s_out=0,
                 eta=1e-5, need_lesser=False, need_greater=False, need_gr_lc=False,
                 keep_gr_left=True):

    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    Wrapper of RGF algorithm to obtain various Green's functions.

    Accepts either a scalar/0-d ``energy`` (legacy callers) or a 1-D ``[B]``
    energy tensor. In the scalar case, all inputs are broadcast to B=1, the
    batched kernel runs, and the leading batch dim is squeezed back out so
    return shapes match the legacy contract. In the batched case, every
    energy-dependent input (``left_se``, ``right_se``, ``s_in``, ``s_out``,
    ``seP``) must already carry the leading ``[B, ...]`` dim, and the
    k-dependent blocks (``hd``, ``sd``, ``hl``, ``hu``, ``sl``, ``su``) may
    arrive 2-D and will be expanded to ``[B, ...]`` zero-copy.

    When every diagonal block shares the same ``n``, the wrapper auto-detects
    and stacks the K blocks into a single ``[K, B, n, n]`` tensor before
    calling the kernel. The K-loop build step then collapses to one fused op,
    the forward solve reuses one cached identity matrix, and per-step list
    overhead in the backward sweeps disappears. Non-uniform geometries fall
    through to the legacy list path. Outputs are unbound back to Python lists
    so downstream callers see the same shape contract either way.

    Parameters
    ----------
    energy : torch.Tensor
        Scalar (0-d) or 1-D ``[B]`` complex tensor.
    s_in : Coupling Matrix Gamma from leads to the device
         (Default value = 0)
    s_out :
         (Default value = 0)
    eta :
         (Default value = 0.000001j)
    need_lesser : bool, optional
        Whether to calculate the lesser Green's function, by default False.
        Lesser Green's function is used for electron density and current density calculation.
    need_greater : bool, optional
        Whether to calculate the greater Green's function, by default False.
        Greater Green's function is used for hole density and phase-breaking scattering case.
    need_gr_lc : bool, optional
        Whether to calculate the left-connected blocks of the retarded Green's function responsible for transmission,
        by default False for memory saving.

    Returns
    -------
     ans: tuple of torch.Tensor
         The output of the recursive Green's function calculation. Leading
         batch dim is squeezed when the caller passed a scalar energy.
    """

    shift_energy = energy + E_ref
    if not torch.is_tensor(shift_energy):
        shift_energy = torch.as_tensor(shift_energy, dtype=torch.complex128)
    # Legacy scalar callers pass either a 0-d tensor or a length-1 1-D tensor together with 2-D Hamiltonian / self-energy inputs.
    # Batched callers pass a 1-D ``[B]`` energy together with 3-D ``[B, n, n]`` tensors.
    # Use the rank of ``left_se`` (or ``right_se``) as the disambiguator so the
    # wrapper can squeeze the batch dim back out for scalar callers.
    # se_probe is used to determine whether the self-energy inputs are 2-D (scalar energy case) or 3-D (batched energy case).
    se_probe = left_se if isinstance(left_se, torch.Tensor) else right_se
    squeezed = isinstance(se_probe, torch.Tensor) and se_probe.ndim == 2
    # if squeezed = True, the wrapper will squeeze the leading batch dim from every output tensor;
    # if False, the wrapper leaves the leading batch dim in place.
    
    if shift_energy.ndim == 0:
        shift_energy = shift_energy.reshape(1)
    elif squeezed and shift_energy.ndim == 1 and shift_energy.shape[0] == 1:
        pass  # already shape [1]; kernel runs B=1, wrapper will squeeze
    B = shift_energy.shape[0]

    def _to_batch(t):
        if t.ndim == 2:
            return t.unsqueeze(0).expand(B, -1, -1)
        return t

    temp_mat_d_list = [_to_batch(hd[i]) * 1. for i in range(len(hd))]
    # L and U are only subtracted out-of-place inside the kernel; the expanded
    # view is fine, and skipping the copy saves K x B x n^2 per list.
    temp_mat_l_list = [_to_batch(hl[i]) for i in range(len(hl))]
    temp_mat_u_list = [_to_batch(hu[i]) for i in range(len(hu))]
    sd_b = [_to_batch(sd[i]) for i in range(len(sd))]
    sl_b = [_to_batch(sl[i]) for i in range(len(sl))]
    su_b = [_to_batch(su[i]) for i in range(len(su))]

    if seP is not None:
        seP_b = [_to_batch(seP[i]) if torch.is_tensor(seP[i]) else seP[i] for i in range(len(seP))]
        for i in range(len(temp_mat_d_list)):
            temp_mat_d_list[i] = temp_mat_d_list[i] + seP_b[i]

    if isinstance(left_se, torch.Tensor):
        left_se_b = _to_batch(left_se)
        s01, s02 = temp_mat_d_list[0].shape[-2], temp_mat_d_list[0].shape[-1]
        se01, se02 = left_se_b.shape[-2], left_se_b.shape[-1]
        idx0, idy0 = min(s01, se01), min(s02, se02)
        temp_mat_d_list[0][:, :idx0, :idy0] = temp_mat_d_list[0][:, :idx0, :idy0] + left_se_b[:, :idx0, :idy0]

    if isinstance(right_se, torch.Tensor):
        right_se_b = _to_batch(right_se)
        s11, s12 = temp_mat_d_list[-1].shape[-2], temp_mat_d_list[-1].shape[-1]
        se11, se12 = right_se_b.shape[-2], right_se_b.shape[-1]
        idx1, idy1 = min(s11, se11), min(s12, se12)
        temp_mat_d_list[-1][:, -idx1:, -idy1:] = temp_mat_d_list[-1][:, -idx1:, -idy1:] + right_se_b[:, -idx1:, -idy1:]

    # s_in / s_out arrive as lists when the lesser/greater paths are active.
    if isinstance(s_in, list):
        s_in_b = [_to_batch(t) if torch.is_tensor(t) else t for t in s_in]
    else:
        s_in_b = s_in
    if isinstance(s_out, list):
        s_out_b = [_to_batch(t) if torch.is_tensor(t) else t for t in s_out]
    else:
        s_out_b = s_out

    # Auto-detect the uniform-block case: every D/L/U block must share the
    # same n x n footprint. Stacking lets the kernel hit a single fused build
    # step and reuse one cached identity across all K forward solves.
    n0 = temp_mat_d_list[0].shape[-1]
    uniform = (
        all(t.shape[-2] == n0 and t.shape[-1] == n0 for t in temp_mat_d_list)
        and all(t.shape[-2] == n0 and t.shape[-1] == n0 for t in temp_mat_l_list)
        and all(t.shape[-2] == n0 and t.shape[-1] == n0 for t in temp_mat_u_list)
    )

    if uniform and len(temp_mat_d_list) >= 2:
        D = torch.stack(temp_mat_d_list, dim=0)       # [K,   B, n, n]
        L = torch.stack(temp_mat_l_list, dim=0)       # [K-1, B, n, n]
        U = torch.stack(temp_mat_u_list, dim=0)       # [K-1, B, n, n]
        Sd = torch.stack(sd_b, dim=0)                 # [K,   B, n, n]
        Sl = torch.stack(sl_b, dim=0)                 # [K-1, B, n, n]
        Su = torch.stack(su_b, dim=0)                 # [K-1, B, n, n]
        ans = recursive_gf_cal(shift_energy, L, D, U, Sd, Su, Sl,
                               s_in=s_in_b, s_out=s_out_b, eta=eta,
                               need_lesser=need_lesser,
                               need_greater=need_greater,
                               need_gr_lc=need_gr_lc,
                               stacked=True,
                               keep_gr_left=keep_gr_left)
    else:
        ans = recursive_gf_cal(shift_energy, temp_mat_l_list, temp_mat_d_list, temp_mat_u_list,
                               sd_b, su_b, sl_b,
                               s_in=s_in_b, s_out=s_out_b, eta=eta,
                               need_lesser=need_lesser,
                               need_greater=need_greater,
                               need_gr_lc=need_gr_lc,
                               stacked=False,
                               keep_gr_left=keep_gr_left)

    if squeezed:
        ans = _squeeze_ans(ans)

    return ans


def _squeeze_ans(ans):
    """Squeeze the leading batch dim from every tensor / list-of-tensors in the
    RGF return tuple so scalar-energy callers see the original 2-D shapes."""
    def _sq(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.squeeze(0)
        if isinstance(x, list):
            return [t.squeeze(0) if torch.is_tensor(t) else t for t in x]
        return x
    return tuple(_sq(x) for x in ans)
