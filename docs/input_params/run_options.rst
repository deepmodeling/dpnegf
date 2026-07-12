===========
Run Options
===========

.. _`run_options`:

.. _`run_op`: 

run_op: 
    | type: ``dict``
    | argument path: ``run_op``

    .. _`run_op/task_options`: 

    task_options: 
        | type: ``dict``, optional
        | argument path: ``run_op/task_options``

        Which transport task to run and its detailed options. See `task_options` below.


        Depending on the value of *task*, different sub args are accepted. 

        .. _`run_op/task_options/task`: 

        task:
            | type: ``str`` (flag key)
            | argument path: ``run_op/task_options/task`` 
            | possible choices: |code:run_op/task_options[negf]|_, |code:run_op/task_options[tbtrans_negf]|_

            The string define the task to conduct:
                                - `negf`: for non-equilibrium green function calculation.
                                - `tbtrans_negf`: for non-equilibrium green function calculation with tbtrans.
                

            .. |code:run_op/task_options[negf]| replace:: ``negf``
            .. _`code:run_op/task_options[negf]`: `run_op/task_options[negf]`_
            .. |code:run_op/task_options[tbtrans_negf]| replace:: ``tbtrans_negf``
            .. _`code:run_op/task_options[tbtrans_negf]`: `run_op/task_options[tbtrans_negf]`_

        .. |flag:run_op/task_options/task| replace:: *task*
        .. _`flag:run_op/task_options/task`: `run_op/task_options/task`_


        .. _`run_op/task_options[negf]`: 

        When |flag:run_op/task_options/task|_ is set to ``negf``: 

        .. _`run_op/task_options[negf]/scf`: 

        scf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/scf``

            Whether to run a self-consistent (Poisson-NEGF) loop. Default: False (non-self-consistent transport).

        .. _`run_op/task_options[negf]/block_tridiagonal`: 

        block_tridiagonal: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/block_tridiagonal``

            Whether to block-tridiagonalize the device Hamiltonian for faster recursive Green's function inversion. Default: False.

        .. _`run_op/task_options[negf]/plot_blocks`: 

        plot_blocks: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/plot_blocks``

            Whether to plot the block tridiagonalization process

        .. _`run_op/task_options[negf]/ele_T`: 

        ele_T: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[negf]/ele_T``

            Electronic temperature in Kelvin. Required.

        .. _`run_op/task_options[negf]/unit`: 

        unit: 
            | type: ``str``, optional, default: ``Hartree``
            | argument path: ``run_op/task_options[negf]/unit``

            Energy unit used for the energy grid and outputs. Choose 'Hartree' (default) or 'eV'.

        .. _`run_op/task_options[negf]/hs_cache`: 

        hs_cache: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/hs_cache``

            On-disk cache of the device Hamiltonian and overlap matrix.

            .. _`run_op/task_options[negf]/hs_cache/use_saved`: 

            use_saved: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/hs_cache/use_saved``

                Whether to load the device Hamiltonian and overlap from an on-disk cache.

            .. _`run_op/task_options[negf]/hs_cache/save_path`: 

            save_path: 
                | type: ``NoneType`` | ``str``, optional, default: ``None``
                | argument path: ``run_op/task_options[negf]/hs_cache/save_path``

                Path to the saved Hamiltonian / overlap file. Required when use_saved is true.

        .. _`run_op/task_options[negf]/scf_options`: 

        scf_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/scf_options``

            SCF mixing options (only used when `scf` is true).


            Depending on the value of *mode*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/scf_options/mode`: 

            mode:
                | type: ``str`` (flag key), default: ``PDIIS``
                | argument path: ``run_op/task_options[negf]/scf_options/mode`` 
                | possible choices: |code:run_op/task_options[negf]/scf_options[PDIIS]|_

                SCF mixing algorithm. Currently only 'PDIIS' is supported.

                .. |code:run_op/task_options[negf]/scf_options[PDIIS]| replace:: ``PDIIS``
                .. _`code:run_op/task_options[negf]/scf_options[PDIIS]`: `run_op/task_options[negf]/scf_options[PDIIS]`_

            .. |flag:run_op/task_options[negf]/scf_options/mode| replace:: *mode*
            .. _`flag:run_op/task_options[negf]/scf_options/mode`: `run_op/task_options[negf]/scf_options/mode`_


            .. _`run_op/task_options[negf]/scf_options[PDIIS]`: 

            When |flag:run_op/task_options[negf]/scf_options/mode|_ is set to ``PDIIS``: 

            PDIIS (Periodic-DIIS) mixing parameters.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/mixing_period`: 

            mixing_period: 
                | type: ``int``, optional, default: ``3``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/mixing_period``

                How many iterations between DIIS extrapolations. Default: 3.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/step_size`: 

            step_size: 
                | type: ``int`` | ``float``, optional, default: ``0.05``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/step_size``

                Linear mixing step size between DIIS steps. Default: 0.05.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/n_history`: 

            n_history: 
                | type: ``int``, optional, default: ``6``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/n_history``

                Number of previous residuals kept for the DIIS subspace. Default: 6.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/abs_err`: 

            abs_err: 
                | type: ``int`` | ``float``, optional, default: ``1e-06``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/abs_err``

                Absolute convergence tolerance on the SCF residual. Default: 1e-6.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/rel_err`: 

            rel_err: 
                | type: ``int`` | ``float``, optional, default: ``0.0001``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/rel_err``

                Relative convergence tolerance on the SCF residual. Default: 1e-4.

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/max_iter``

                Maximum SCF iterations. Default: 100.

        .. _`run_op/task_options[negf]/stru_options`: 

        stru_options: 
            | type: ``dict``
            | argument path: ``run_op/task_options[negf]/stru_options``

            Structure partition and k-mesh options: device region, left/right leads, PBC, k-mesh.

            .. _`run_op/task_options[negf]/stru_options/device`: 

            device: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/device``

                Device region: atom index range and sorting.

                .. _`run_op/task_options[negf]/stru_options/device/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/device/id``

                    Atom index range of the device region, e.g. '32-64' (0-indexed, half-open).

                .. _`run_op/task_options[negf]/stru_options/device/sort`: 

                sort: 
                    | type: ``bool``, optional, default: ``True``
                    | argument path: ``run_op/task_options[negf]/stru_options/device/sort``

                    Whether to sort atoms in the device region along the transport axis. Default: True.

            .. _`run_op/task_options[negf]/stru_options/lead_L`: 

            lead_L: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/lead_L``

                Left lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion.

                .. _`run_op/task_options[negf]/stru_options/lead_L/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/id``

                    Atom index range of the lead region, e.g. '0-32' (0-indexed, half-open).

                .. _`run_op/task_options[negf]/stru_options/lead_L/voltage`: 

                voltage: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/voltage``

                    Applied bias voltage on this lead, in `unit`. Required.

                .. _`run_op/task_options[negf]/stru_options/lead_L/useBloch`: 

                useBloch: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/useBloch``

                    Whether to use Bloch expansion to reduce lead unit cell size. Default: False.

                .. _`run_op/task_options[negf]/stru_options/lead_L/bloch_factor`: 

                bloch_factor: 
                    | type: ``list``, optional, default: ``[1, 1, 1]``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/bloch_factor``

                    Bloch factor [nx, ny, nz] for the lead unit cell. Default: [1, 1, 1].

                .. _`run_op/task_options[negf]/stru_options/lead_L/kmesh_lead_Ef`: 

                kmesh_lead_Ef: 
                    | type: ``list``, optional
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/kmesh_lead_Ef``

                    K-mesh used to compute the lead Fermi level, [nkx, nky, nkz].

                .. _`run_op/task_options[negf]/stru_options/lead_L/charge`: 

                charge: 
                    | type: ``int`` | ``float``, optional, default: ``0.0``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/charge``

                    Doping charge on the lead, used when computing the lead Fermi level. Default: 0.0.

            .. _`run_op/task_options[negf]/stru_options/lead_R`: 

            lead_R: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/lead_R``

                Right lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion.

                .. _`run_op/task_options[negf]/stru_options/lead_R/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/id``

                    Atom index range of the lead region, e.g. '0-32' (0-indexed, half-open).

                .. _`run_op/task_options[negf]/stru_options/lead_R/voltage`: 

                voltage: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/voltage``

                    Applied bias voltage on this lead, in `unit`. Required.

                .. _`run_op/task_options[negf]/stru_options/lead_R/useBloch`: 

                useBloch: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/useBloch``

                    Whether to use Bloch expansion to reduce lead unit cell size. Default: False.

                .. _`run_op/task_options[negf]/stru_options/lead_R/bloch_factor`: 

                bloch_factor: 
                    | type: ``list``, optional, default: ``[1, 1, 1]``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/bloch_factor``

                    Bloch factor [nx, ny, nz] for the lead unit cell. Default: [1, 1, 1].

                .. _`run_op/task_options[negf]/stru_options/lead_R/kmesh_lead_Ef`: 

                kmesh_lead_Ef: 
                    | type: ``list``, optional
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/kmesh_lead_Ef``

                    K-mesh used to compute the lead Fermi level, [nkx, nky, nkz].

                .. _`run_op/task_options[negf]/stru_options/lead_R/charge`: 

                charge: 
                    | type: ``int`` | ``float``, optional, default: ``0.0``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/charge``

                    Doping charge on the lead, used when computing the lead Fermi level. Default: 0.0.

            .. _`run_op/task_options[negf]/stru_options/kmesh`: 

            kmesh: 
                | type: ``list``, optional, default: ``[1, 1, 1]``
                | argument path: ``run_op/task_options[negf]/stru_options/kmesh``

                K-mesh used for the device transverse Brillouin-zone sampling, [nkx, nky, nkz]. Default: [1, 1, 1].

            .. _`run_op/task_options[negf]/stru_options/pbc`: 

            pbc: 
                | type: ``list``, optional, default: ``[False, False, False]``
                | argument path: ``run_op/task_options[negf]/stru_options/pbc``

                Periodic boundary condition applied to the device, list of three booleans for x/y/z. Default: [False, False, False].

            .. _`run_op/task_options[negf]/stru_options/gamma_center`: 

            gamma_center: 
                | type: ``bool`` | ``list``, optional, default: ``True``
                | argument path: ``run_op/task_options[negf]/stru_options/gamma_center``

                Whether the k-mesh is Gamma-centered. Default: True.

            .. _`run_op/task_options[negf]/stru_options/time_reversal_symmetry`: 

            time_reversal_symmetry: 
                | type: ``bool`` | ``list``, optional, default: ``True``
                | argument path: ``run_op/task_options[negf]/stru_options/time_reversal_symmetry``

                Whether to enforce time-reversal symmetry when folding the k-mesh. Default: True.

            .. _`run_op/task_options[negf]/stru_options/e_fermi_smearing`: 

            e_fermi_smearing: 
                | type: ``str``, optional, default: ``FD``
                | argument path: ``run_op/task_options[negf]/stru_options/e_fermi_smearing``

                Smearing method for lead Fermi-level determination. Default: 'FD' (Fermi-Dirac).

            .. _`run_op/task_options[negf]/stru_options/eig_solver`: 

            eig_solver: 
                | type: ``str``, optional, default: ``torch``
                | argument path: ``run_op/task_options[negf]/stru_options/eig_solver``

                Eigenvalue solver used for lead Fermi-level calculation. Default: 'torch'.

            .. _`run_op/task_options[negf]/stru_options/nel_atom`: 

            nel_atom: 
                | type: ``NoneType`` | ``dict``, optional, default: ``None``
                | argument path: ``run_op/task_options[negf]/stru_options/nel_atom``

                Number of valence electrons per element, e.g. {'C': 4}. Required for SCF or when computing lead Fermi-level.

        .. _`run_op/task_options[negf]/poisson_options`: 

        poisson_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/poisson_options``

            Poisson solver options (only used when `scf` is true).


            Depending on the value of *solver*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/poisson_options/solver`: 

            solver:
                | type: ``str`` (flag key), default: ``fmm``
                | argument path: ``run_op/task_options[negf]/poisson_options/solver`` 
                | possible choices: |code:run_op/task_options[negf]/poisson_options[fmm]|_, |code:run_op/task_options[negf]/poisson_options[pyamg]|_, |code:run_op/task_options[negf]/poisson_options[scipy]|_

                Poisson solver backend. 'fmm' (default) uses the free-space Fast Multipole Method; 'pyamg' uses an algebraic-multigrid finite-difference solver on a grid; 'scipy' uses SciPy's sparse linear algebra on a grid.

                .. |code:run_op/task_options[negf]/poisson_options[fmm]| replace:: ``fmm``
                .. _`code:run_op/task_options[negf]/poisson_options[fmm]`: `run_op/task_options[negf]/poisson_options[fmm]`_
                .. |code:run_op/task_options[negf]/poisson_options[pyamg]| replace:: ``pyamg``
                .. _`code:run_op/task_options[negf]/poisson_options[pyamg]`: `run_op/task_options[negf]/poisson_options[pyamg]`_
                .. |code:run_op/task_options[negf]/poisson_options[scipy]| replace:: ``scipy``
                .. _`code:run_op/task_options[negf]/poisson_options[scipy]`: `run_op/task_options[negf]/poisson_options[scipy]`_

            .. |flag:run_op/task_options[negf]/poisson_options/solver| replace:: *solver*
            .. _`flag:run_op/task_options[negf]/poisson_options/solver`: `run_op/task_options[negf]/poisson_options/solver`_


            .. _`run_op/task_options[negf]/poisson_options[fmm]`: 

            When |flag:run_op/task_options[negf]/poisson_options/solver|_ is set to ``fmm``: 

            FMM solver options.

            .. _`run_op/task_options[negf]/poisson_options[fmm]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[negf]/poisson_options[fmm]/err``

                Requested FMM accuracy on the Coulomb sum. Default: 1e-5.


            .. _`run_op/task_options[negf]/poisson_options[pyamg]`: 

            When |flag:run_op/task_options[negf]/poisson_options/solver|_ is set to ``pyamg``: 

            pyamg (algebraic multigrid) solver options: grid, gates, dielectric and doped regions.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/err``

                Error target passed to the AMG solver. Default: 1e-5.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/tolerance`: 

            tolerance: 
                | type: ``int`` | ``float``, optional, default: ``1e-07``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/tolerance``

                Convergence tolerance of the AMG linear solve. Default: 1e-7.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/max_iter``

                Maximum AMG iterations. Default: 100.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/mix_rate`: 

            mix_rate: 
                | type: ``int`` | ``float``, optional, default: ``0.25``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/mix_rate``

                Underrelaxation between successive Poisson solves. Default: 0.25.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/poisson_dtype`: 

            poisson_dtype: 
                | type: ``str``, optional, default: ``float64``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/poisson_dtype``

                Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/grid`: 

            grid: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/grid``

                Real-space grid used to discretize the Poisson equation.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/grid/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/grid/x_range``

                    Grid range along x as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/grid/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/grid/y_range``

                    Grid range along y as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/grid/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/grid/z_range``

                    Grid range along z as 'start:stop:n' (Angstrom, n points).

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_top`: 

            gate_top: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_top``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_top/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_top/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_top/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_top/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_top/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_top/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_top/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_top/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom`: 

            gate_bottom: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_bottom/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_left`: 

            gate_left: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_left``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_left/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_left/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_left/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_left/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_left/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_left/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_left/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_left/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_right`: 

            gate_right: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_right``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_right/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_right/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_right/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_right/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_right/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_right/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/gate_right/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/gate_right/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region`: 

            dielectric_region: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2`: 

            dielectric_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region2/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3`: 

            dielectric_region3: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region3/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4`: 

            dielectric_region4: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region4/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5`: 

            dielectric_region5: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region5/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6`: 

            dielectric_region6: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/dielectric_region6/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[pyamg]/doped_region`: 

            doped_region: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/doped_region``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/doped_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/doped_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/doped_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/doped_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/doped_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/doped_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[pyamg]/doped_region/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[pyamg]/doped_region/charge``

                    Fixed doping charge (electrons per unit cell of the region).


            .. _`run_op/task_options[negf]/poisson_options[scipy]`: 

            When |flag:run_op/task_options[negf]/poisson_options/solver|_ is set to ``scipy``: 

            SciPy sparse solver options: grid, gates, dielectric and doped regions.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/err``

                Residual threshold on the potential update. Default: 1e-5.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/tolerance`: 

            tolerance: 
                | type: ``int`` | ``float``, optional, default: ``1e-07``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/tolerance``

                Convergence tolerance of the sparse linear solve. Default: 1e-7.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/max_iter``

                Maximum Poisson iterations. Default: 100.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/mix_rate`: 

            mix_rate: 
                | type: ``int`` | ``float``, optional, default: ``0.25``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/mix_rate``

                Underrelaxation between successive Poisson solves. Default: 0.25.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/poisson_dtype`: 

            poisson_dtype: 
                | type: ``str``, optional, default: ``float64``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/poisson_dtype``

                Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/with_Dirichlet_leads`: 

            with_Dirichlet_leads: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/with_Dirichlet_leads``

                Whether to use Dirichlet boundary condition for leads

            .. _`run_op/task_options[negf]/poisson_options[scipy]/grid`: 

            grid: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/grid``

                Real-space grid used to discretize the Poisson equation.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/grid/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/grid/x_range``

                    Grid range along x as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/grid/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/grid/y_range``

                    Grid range along y as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/grid/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/grid/z_range``

                    Grid range along z as 'start:stop:n' (Angstrom, n points).

            .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_top`: 

            gate_top: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_top``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_top/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_top/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_top/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_top/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_top/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_top/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_top/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_top/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_bottom`: 

            gate_bottom: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_bottom``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_bottom/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_left`: 

            gate_left: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_left``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_left/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_left/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_left/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_left/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_left/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_left/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_left/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_left/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_right`: 

            gate_right: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_right``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_right/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_right/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_right/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_right/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_right/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_right/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/gate_right/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/gate_right/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_L`: 

            lead_L: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_L``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_L/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_L/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_L/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_L/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_L/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_L/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_L/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_L/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_R`: 

            lead_R: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_R``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_R/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_R/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_R/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_R/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_R/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_R/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/lead_R/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/lead_R/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region`: 

            dielectric_region: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2`: 

            dielectric_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region2/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3`: 

            dielectric_region3: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region3/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4`: 

            dielectric_region4: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region4/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5`: 

            dielectric_region5: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region5/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6`: 

            dielectric_region6: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/dielectric_region6/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region1`: 

            doped_region1: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region1``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region1/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region1/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region1/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region1/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region1/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region1/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region1/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region1/charge``

                    Fixed doping charge (electrons per unit cell of the region).

            .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region2`: 

            doped_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region2``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[negf]/poisson_options[scipy]/doped_region2/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[negf]/poisson_options[scipy]/doped_region2/charge``

                    Fixed doping charge (electrons per unit cell of the region).

        .. _`run_op/task_options[negf]/self_energy_options`: 

        self_energy_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/self_energy_options``

            Self-energy stage options: SGF solver, numba JIT, on-disk cache, and CPU parallelism (joblib workers, BLAS threads, ek batching).

            .. _`run_op/task_options[negf]/self_energy_options/solver`: 

            solver: 
                | type: ``str``, optional, default: ``Sancho-Rubio``
                | argument path: ``run_op/task_options[negf]/self_energy_options/solver``

                Surface Green's function solver. 'Sancho-Rubio' (default) or 'Lopez-Sancho'.

            .. _`run_op/task_options[negf]/self_energy_options/numba_jit`: 

            numba_jit: 
                | type: ``bool`` | ``NoneType``, optional, default: ``None``
                | argument path: ``run_op/task_options[negf]/self_energy_options/numba_jit``

                Whether to JIT-compile the surface Green's function core with numba. None (default) uses numba when available.

            .. _`run_op/task_options[negf]/self_energy_options/info_display`: 

            info_display: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/self_energy_options/info_display``

                Whether to log per-(k, E) self-energy solver info. Verbose; off by default.

            .. _`run_op/task_options[negf]/self_energy_options/cache`: 

            cache: 
                | type: ``dict``, optional, default: ``{}``
                | argument path: ``run_op/task_options[negf]/self_energy_options/cache``

                On-disk cache of the self-energy for reuse across runs.

                .. _`run_op/task_options[negf]/self_energy_options/cache/use_saved`: 

                use_saved: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/cache/use_saved``

                    Whether to load self-energy from an on-disk HDF5 cache instead of recomputing.

                .. _`run_op/task_options[negf]/self_energy_options/cache/save_path`: 

                save_path: 
                    | type: ``NoneType`` | ``str``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/cache/save_path``

                    Directory that holds (or will hold) the self-energy HDF5 files. If None, defaults to `<results_path>/self_energy`.

            .. _`run_op/task_options[negf]/self_energy_options/parallel`: 

            parallel: 
                | type: ``dict``, optional, default: ``{}``
                | argument path: ``run_op/task_options[negf]/self_energy_options/parallel``

                CPU parallelism knobs for the self-energy sweep.

                .. _`run_op/task_options[negf]/self_energy_options/parallel/n_workers`: 

                n_workers: 
                    | type: ``int`` | ``NoneType``, optional, default: ``-1``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/parallel/n_workers``

                    Number of joblib workers used to parallelize the self-energy sweep over (k, E). -1 (default) auto-selects based on the CPU and memory budget (see `_get_safe_n_jobs`).

                .. _`run_op/task_options[negf]/self_energy_options/parallel/cpu_budget`: 

                cpu_budget: 
                    | type: ``int`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/parallel/cpu_budget``

                    Total CPU cores the self-energy pool is allowed to size against. None (default) uses os.cpu_count(). This bounds both `n_workers` and `blas_threads` so BLAS-thread * worker <= cpu_budget.

                .. _`run_op/task_options[negf]/self_energy_options/parallel/blas_threads`: 

                blas_threads: 
                    | type: ``int`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/parallel/blas_threads``

                    BLAS/LAPACK threads per worker inside the Lopez-Sancho loop. None (default) auto-tunes by timing a sample (k, E) at 1, 2, 4, 8, and cpu_budget/n_workers threads and keeping the fastest.

                .. _`run_op/task_options[negf]/self_energy_options/parallel/ek_batch_size`: 

                ek_batch_size: 
                    | type: ``int``, optional, default: ``200``
                    | argument path: ``run_op/task_options[negf]/self_energy_options/parallel/ek_batch_size``

                    Batch size for (k, E) tasks handed to joblib. Larger = fewer batches, more memory. Default 200.

        .. _`run_op/task_options[negf]/espacing`: 

        espacing: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[negf]/espacing``

            Energy grid spacing in `unit`. Required.

        .. _`run_op/task_options[negf]/emin`: 

        emin: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[negf]/emin``

            Lower bound of the energy grid in `unit`. Required.

        .. _`run_op/task_options[negf]/emax`: 

        emax: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[negf]/emax``

            Upper bound of the energy grid in `unit`. Required.

        .. _`run_op/task_options[negf]/rgf_options`: 

        rgf_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/rgf_options``

            Recursive Green's function stage options: compute device ('cpu' or 'cuda') and energy-loop chunk size. Independent from the self-energy pool, which always runs on CPU.

            .. _`run_op/task_options[negf]/rgf_options/device`: 

            device: 
                | type: ``str``, optional, default: ``cpu``
                | argument path: ``run_op/task_options[negf]/rgf_options/device``

                Device used only for the RGF (recursive Green's function) step. 'cpu' (default) or 'cuda'. Hamiltonian initialization always runs on CPU (it is the memory-heavy phase). Self-energy always runs on CPU because it goes through joblib + numba.

            .. _`run_op/task_options[negf]/rgf_options/e_batch_size`: 

            e_batch_size: 
                | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                | argument path: ``run_op/task_options[negf]/rgf_options/e_batch_size``

                Number of energy points solved in one batched RGF call. None (default) auto-picks from free CUDA memory on 'cuda'; on 'cpu' the full grid is used.

        .. _`run_op/task_options[negf]/e_fermi`: 

        e_fermi: 
            | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[negf]/e_fermi``

            Device Fermi level in `unit`. Optional; if None, computed from the lead Fermi level and applied to the device.

        .. _`run_op/task_options[negf]/density_options`: 

        density_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/density_options``


            Depending on the value of *method*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/density_options/method`: 

            method:
                | type: ``str`` (flag key), default: ``Ozaki``
                | argument path: ``run_op/task_options[negf]/density_options/method`` 
                | possible choices: |code:run_op/task_options[negf]/density_options[Ozaki]|_, |code:run_op/task_options[negf]/density_options[Fiori]|_

                Density integration method. 'Ozaki' (default) uses the Ozaki contour with Matsubara-pole expansion; 'Fiori' uses real-axis integration with Gauss quadrature (used with the Fiori/effective-mass approach).

                .. |code:run_op/task_options[negf]/density_options[Ozaki]| replace:: ``Ozaki``
                .. _`code:run_op/task_options[negf]/density_options[Ozaki]`: `run_op/task_options[negf]/density_options[Ozaki]`_
                .. |code:run_op/task_options[negf]/density_options[Fiori]| replace:: ``Fiori``
                .. _`code:run_op/task_options[negf]/density_options[Fiori]`: `run_op/task_options[negf]/density_options[Fiori]`_

            .. |flag:run_op/task_options[negf]/density_options/method| replace:: *method*
            .. _`flag:run_op/task_options[negf]/density_options/method`: `run_op/task_options[negf]/density_options/method`_


            .. _`run_op/task_options[negf]/density_options[Ozaki]`: 

            When |flag:run_op/task_options[negf]/density_options/method|_ is set to ``Ozaki``: 

            Ozaki-contour options.

            .. _`run_op/task_options[negf]/density_options[Ozaki]/R`: 

            R: 
                | type: ``int`` | ``float``, optional, default: ``1000000.0``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/R``

                Radius of the semicircular contour in `unit`. Default: 1e6.

            .. _`run_op/task_options[negf]/density_options[Ozaki]/M_cut`: 

            M_cut: 
                | type: ``int``, optional, default: ``30``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/M_cut``

                Number of Ozaki poles kept. Default: 30.

            .. _`run_op/task_options[negf]/density_options[Ozaki]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``10``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/n_gauss``

                Number of Gauss-Legendre points on the equilibrium contour. Default: 10.


            .. _`run_op/task_options[negf]/density_options[Fiori]`: 

            When |flag:run_op/task_options[negf]/density_options/method|_ is set to ``Fiori``: 

            Fiori real-axis integration options.

            .. _`run_op/task_options[negf]/density_options[Fiori]/integrate_way`: 

            integrate_way: 
                | type: ``int`` | ``str``, optional, default: ``direct``
                | argument path: ``run_op/task_options[negf]/density_options[Fiori]/integrate_way``

                Integration strategy for the Fiori method: 'direct' or 'gauss'. Default: 'direct'.

            .. _`run_op/task_options[negf]/density_options[Fiori]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[negf]/density_options[Fiori]/n_gauss``

                Number of Gauss quadrature points along the real-axis integration. Default: 100.

        .. _`run_op/task_options[negf]/eta_lead`: 

        eta_lead: 
            | type: ``int`` | ``float``, optional, default: ``1e-05``
            | argument path: ``run_op/task_options[negf]/eta_lead``

            Small imaginary broadening added to the lead energy. Default: 1e-5.

        .. _`run_op/task_options[negf]/eta_device`: 

        eta_device: 
            | type: ``int`` | ``float``, optional, default: ``0.0``
            | argument path: ``run_op/task_options[negf]/eta_device``

            Small imaginary broadening added to the device energy. Default: 0.

        .. _`run_op/task_options[negf]/output_options`: 

        output_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/output_options``

            Which physical quantities to write out at the end of the run.

            .. _`run_op/task_options[negf]/output_options/dos`: 

            dos: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/dos``

                Write density of states.

            .. _`run_op/task_options[negf]/output_options/tc`: 

            tc: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/tc``

                Write transmission coefficient.

            .. _`run_op/task_options[negf]/output_options/density`: 

            density: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/density``

                Write electron density.

            .. _`run_op/task_options[negf]/output_options/potential`: 

            potential: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/potential``

                Write electrostatic potential.

            .. _`run_op/task_options[negf]/output_options/current`: 

            current: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/current``

                Write self-consistent current.

            .. _`run_op/task_options[negf]/output_options/current_nscf`: 

            current_nscf: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/current_nscf``

                Write non-self-consistent current.

            .. _`run_op/task_options[negf]/output_options/ldos`: 

            ldos: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/ldos``

                Write local density of states.

            .. _`run_op/task_options[negf]/output_options/lcurrent`: 

            lcurrent: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[negf]/output_options/lcurrent``

                Write local current.


        .. _`run_op/task_options[tbtrans_negf]`: 

        When |flag:run_op/task_options/task|_ is set to ``tbtrans_negf``: 

        .. _`run_op/task_options[tbtrans_negf]/scf`: 

        scf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/scf``

            Whether to run a self-consistent (Poisson-NEGF) loop. Default: False (non-self-consistent transport).

        .. _`run_op/task_options[tbtrans_negf]/block_tridiagonal`: 

        block_tridiagonal: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/block_tridiagonal``

            Whether to block-tridiagonalize the device Hamiltonian for faster recursive Green's function inversion. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/ele_T`: 

        ele_T: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[tbtrans_negf]/ele_T``

            Electronic temperature in Kelvin. Required.

        .. _`run_op/task_options[tbtrans_negf]/unit`: 

        unit: 
            | type: ``str``, optional, default: ``Hartree``
            | argument path: ``run_op/task_options[tbtrans_negf]/unit``

            Energy unit used for the energy grid and outputs. Choose 'Hartree' (default) or 'eV'.

        .. _`run_op/task_options[tbtrans_negf]/scf_options`: 

        scf_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/scf_options``

            SCF mixing options (only used when `scf` is true).


            Depending on the value of *mode*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/scf_options/mode`: 

            mode:
                | type: ``str`` (flag key), default: ``PDIIS``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options/mode`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]|_

                SCF mixing algorithm. Currently only 'PDIIS' is supported.

                .. |code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]| replace:: ``PDIIS``
                .. _`code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`: `run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`_

            .. |flag:run_op/task_options[tbtrans_negf]/scf_options/mode| replace:: *mode*
            .. _`flag:run_op/task_options[tbtrans_negf]/scf_options/mode`: `run_op/task_options[tbtrans_negf]/scf_options/mode`_


            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`: 

            When |flag:run_op/task_options[tbtrans_negf]/scf_options/mode|_ is set to ``PDIIS``: 

            PDIIS (Periodic-DIIS) mixing parameters.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/mixing_period`: 

            mixing_period: 
                | type: ``int``, optional, default: ``3``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/mixing_period``

                How many iterations between DIIS extrapolations. Default: 3.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/step_size`: 

            step_size: 
                | type: ``int`` | ``float``, optional, default: ``0.05``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/step_size``

                Linear mixing step size between DIIS steps. Default: 0.05.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/n_history`: 

            n_history: 
                | type: ``int``, optional, default: ``6``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/n_history``

                Number of previous residuals kept for the DIIS subspace. Default: 6.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/abs_err`: 

            abs_err: 
                | type: ``int`` | ``float``, optional, default: ``1e-06``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/abs_err``

                Absolute convergence tolerance on the SCF residual. Default: 1e-6.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/rel_err`: 

            rel_err: 
                | type: ``int`` | ``float``, optional, default: ``0.0001``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/rel_err``

                Relative convergence tolerance on the SCF residual. Default: 1e-4.

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/max_iter``

                Maximum SCF iterations. Default: 100.

        .. _`run_op/task_options[tbtrans_negf]/stru_options`: 

        stru_options: 
            | type: ``dict``
            | argument path: ``run_op/task_options[tbtrans_negf]/stru_options``

            Structure partition and k-mesh options: device region, left/right leads, PBC, k-mesh.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/device`: 

            device: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device``

                Device region: atom index range and sorting.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/device/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device/id``

                    Atom index range of the device region, e.g. '32-64' (0-indexed, half-open).

                .. _`run_op/task_options[tbtrans_negf]/stru_options/device/sort`: 

                sort: 
                    | type: ``bool``, optional, default: ``True``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device/sort``

                    Whether to sort atoms in the device region along the transport axis. Default: True.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L`: 

            lead_L: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L``

                Left lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/id``

                    Atom index range of the lead region, e.g. '0-32' (0-indexed, half-open).

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/voltage`: 

                voltage: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/voltage``

                    Applied bias voltage on this lead, in `unit`. Required.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/useBloch`: 

                useBloch: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/useBloch``

                    Whether to use Bloch expansion to reduce lead unit cell size. Default: False.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/bloch_factor`: 

                bloch_factor: 
                    | type: ``list``, optional, default: ``[1, 1, 1]``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/bloch_factor``

                    Bloch factor [nx, ny, nz] for the lead unit cell. Default: [1, 1, 1].

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/kmesh_lead_Ef`: 

                kmesh_lead_Ef: 
                    | type: ``list``, optional
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/kmesh_lead_Ef``

                    K-mesh used to compute the lead Fermi level, [nkx, nky, nkz].

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/charge`: 

                charge: 
                    | type: ``int`` | ``float``, optional, default: ``0.0``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/charge``

                    Doping charge on the lead, used when computing the lead Fermi level. Default: 0.0.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R`: 

            lead_R: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R``

                Right lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/id``

                    Atom index range of the lead region, e.g. '0-32' (0-indexed, half-open).

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/voltage`: 

                voltage: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/voltage``

                    Applied bias voltage on this lead, in `unit`. Required.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/useBloch`: 

                useBloch: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/useBloch``

                    Whether to use Bloch expansion to reduce lead unit cell size. Default: False.

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/bloch_factor`: 

                bloch_factor: 
                    | type: ``list``, optional, default: ``[1, 1, 1]``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/bloch_factor``

                    Bloch factor [nx, ny, nz] for the lead unit cell. Default: [1, 1, 1].

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/kmesh_lead_Ef`: 

                kmesh_lead_Ef: 
                    | type: ``list``, optional
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/kmesh_lead_Ef``

                    K-mesh used to compute the lead Fermi level, [nkx, nky, nkz].

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/charge`: 

                charge: 
                    | type: ``int`` | ``float``, optional, default: ``0.0``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/charge``

                    Doping charge on the lead, used when computing the lead Fermi level. Default: 0.0.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/kmesh`: 

            kmesh: 
                | type: ``list``, optional, default: ``[1, 1, 1]``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/kmesh``

                K-mesh used for the device transverse Brillouin-zone sampling, [nkx, nky, nkz]. Default: [1, 1, 1].

            .. _`run_op/task_options[tbtrans_negf]/stru_options/pbc`: 

            pbc: 
                | type: ``list``, optional, default: ``[False, False, False]``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/pbc``

                Periodic boundary condition applied to the device, list of three booleans for x/y/z. Default: [False, False, False].

            .. _`run_op/task_options[tbtrans_negf]/stru_options/gamma_center`: 

            gamma_center: 
                | type: ``bool`` | ``list``, optional, default: ``True``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/gamma_center``

                Whether the k-mesh is Gamma-centered. Default: True.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/time_reversal_symmetry`: 

            time_reversal_symmetry: 
                | type: ``bool`` | ``list``, optional, default: ``True``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/time_reversal_symmetry``

                Whether to enforce time-reversal symmetry when folding the k-mesh. Default: True.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/e_fermi_smearing`: 

            e_fermi_smearing: 
                | type: ``str``, optional, default: ``FD``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/e_fermi_smearing``

                Smearing method for lead Fermi-level determination. Default: 'FD' (Fermi-Dirac).

            .. _`run_op/task_options[tbtrans_negf]/stru_options/eig_solver`: 

            eig_solver: 
                | type: ``str``, optional, default: ``torch``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/eig_solver``

                Eigenvalue solver used for lead Fermi-level calculation. Default: 'torch'.

            .. _`run_op/task_options[tbtrans_negf]/stru_options/nel_atom`: 

            nel_atom: 
                | type: ``NoneType`` | ``dict``, optional, default: ``None``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/nel_atom``

                Number of valence electrons per element, e.g. {'C': 4}. Required for SCF or when computing lead Fermi-level.

        .. _`run_op/task_options[tbtrans_negf]/poisson_options`: 

        poisson_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options``

            Poisson solver options (only used when `scf` is true).


            Depending on the value of *solver*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/poisson_options/solver`: 

            solver:
                | type: ``str`` (flag key), default: ``fmm``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options/solver`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]|_, |code:run_op/task_options[tbtrans_negf]/poisson_options[pyamg]|_, |code:run_op/task_options[tbtrans_negf]/poisson_options[scipy]|_

                Poisson solver backend. 'fmm' (default) uses the free-space Fast Multipole Method; 'pyamg' uses an algebraic-multigrid finite-difference solver on a grid; 'scipy' uses SciPy's sparse linear algebra on a grid.

                .. |code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]| replace:: ``fmm``
                .. _`code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]`: `run_op/task_options[tbtrans_negf]/poisson_options[fmm]`_
                .. |code:run_op/task_options[tbtrans_negf]/poisson_options[pyamg]| replace:: ``pyamg``
                .. _`code:run_op/task_options[tbtrans_negf]/poisson_options[pyamg]`: `run_op/task_options[tbtrans_negf]/poisson_options[pyamg]`_
                .. |code:run_op/task_options[tbtrans_negf]/poisson_options[scipy]| replace:: ``scipy``
                .. _`code:run_op/task_options[tbtrans_negf]/poisson_options[scipy]`: `run_op/task_options[tbtrans_negf]/poisson_options[scipy]`_

            .. |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver| replace:: *solver*
            .. _`flag:run_op/task_options[tbtrans_negf]/poisson_options/solver`: `run_op/task_options[tbtrans_negf]/poisson_options/solver`_


            .. _`run_op/task_options[tbtrans_negf]/poisson_options[fmm]`: 

            When |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver|_ is set to ``fmm``: 

            FMM solver options.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[fmm]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[fmm]/err``

                Requested FMM accuracy on the Coulomb sum. Default: 1e-5.


            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]`: 

            When |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver|_ is set to ``pyamg``: 

            pyamg (algebraic multigrid) solver options: grid, gates, dielectric and doped regions.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/err``

                Error target passed to the AMG solver. Default: 1e-5.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/tolerance`: 

            tolerance: 
                | type: ``int`` | ``float``, optional, default: ``1e-07``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/tolerance``

                Convergence tolerance of the AMG linear solve. Default: 1e-7.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/max_iter``

                Maximum AMG iterations. Default: 100.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/mix_rate`: 

            mix_rate: 
                | type: ``int`` | ``float``, optional, default: ``0.25``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/mix_rate``

                Underrelaxation between successive Poisson solves. Default: 0.25.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/poisson_dtype`: 

            poisson_dtype: 
                | type: ``str``, optional, default: ``float64``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/poisson_dtype``

                Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid`: 

            grid: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid``

                Real-space grid used to discretize the Poisson equation.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/x_range``

                    Grid range along x as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/y_range``

                    Grid range along y as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/grid/z_range``

                    Grid range along z as 'start:stop:n' (Angstrom, n points).

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top`: 

            gate_top: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_top/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom`: 

            gate_bottom: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_bottom/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left`: 

            gate_left: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_left/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right`: 

            gate_right: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/gate_right/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region`: 

            dielectric_region: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2`: 

            dielectric_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region2/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3`: 

            dielectric_region3: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region3/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4`: 

            dielectric_region4: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region4/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5`: 

            dielectric_region5: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region5/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6`: 

            dielectric_region6: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/dielectric_region6/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region`: 

            doped_region: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[pyamg]/doped_region/charge``

                    Fixed doping charge (electrons per unit cell of the region).


            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]`: 

            When |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver|_ is set to ``scipy``: 

            SciPy sparse solver options: grid, gates, dielectric and doped regions.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/err`: 

            err: 
                | type: ``int`` | ``float``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/err``

                Residual threshold on the potential update. Default: 1e-5.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/tolerance`: 

            tolerance: 
                | type: ``int`` | ``float``, optional, default: ``1e-07``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/tolerance``

                Convergence tolerance of the sparse linear solve. Default: 1e-7.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/max_iter``

                Maximum Poisson iterations. Default: 100.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/mix_rate`: 

            mix_rate: 
                | type: ``int`` | ``float``, optional, default: ``0.25``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/mix_rate``

                Underrelaxation between successive Poisson solves. Default: 0.25.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/poisson_dtype`: 

            poisson_dtype: 
                | type: ``str``, optional, default: ``float64``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/poisson_dtype``

                Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/with_Dirichlet_leads`: 

            with_Dirichlet_leads: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/with_Dirichlet_leads``

                Whether to use Dirichlet boundary condition for leads

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid`: 

            grid: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid``

                Real-space grid used to discretize the Poisson equation.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/x_range``

                    Grid range along x as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/y_range``

                    Grid range along y as 'start:stop:n' (Angstrom, n points).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/grid/z_range``

                    Grid range along z as 'start:stop:n' (Angstrom, n points).

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top`: 

            gate_top: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_top/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom`: 

            gate_bottom: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_bottom/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left`: 

            gate_left: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_left/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right`: 

            gate_right: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/gate_right/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L`: 

            lead_L: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_L/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R`: 

            lead_R: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R``

                Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/voltage`: 

                voltage: 
                    | type: ``int`` | ``float`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/lead_R/voltage``

                    Applied voltage on this Dirichlet boundary in `unit`. Default: None.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region`: 

            dielectric_region: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2`: 

            dielectric_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region2/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3`: 

            dielectric_region3: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region3/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4`: 

            dielectric_region4: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region4/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5`: 

            dielectric_region5: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region5/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6`: 

            dielectric_region6: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6``

                Rectangular region with a fixed relative permittivity. Additional regions can be added via `dielectric_region2` ... `dielectric_region6`.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/relative permittivity`: 

                relative permittivity: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/dielectric_region6/relative permittivity``

                    Relative permittivity of the region.

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1`: 

            doped_region1: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region1/charge``

                    Fixed doping charge (electrons per unit cell of the region).

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2`: 

            doped_region2: 
                | type: ``dict``, optional
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2``

                Rectangular region with a fixed doping charge density.

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/x_range`: 

                x_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/x_range``

                    Range along x as 'x0:x1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/y_range`: 

                y_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/y_range``

                    Range along y as 'y0:y1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/z_range`: 

                z_range: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/z_range``

                    Range along z as 'z0:z1' (Angstrom).

                .. _`run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/charge`: 

                charge: 
                    | type: ``int`` | ``float``
                    | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[scipy]/doped_region2/charge``

                    Fixed doping charge (electrons per unit cell of the region).

        .. _`run_op/task_options[tbtrans_negf]/sgf_solver`: 

        sgf_solver: 
            | type: ``str``, optional, default: ``Sancho-Rubio``
            | argument path: ``run_op/task_options[tbtrans_negf]/sgf_solver``

            Surface Green's function solver. 'Sancho-Rubio' (default) or 'Lopez-Sancho'.

        .. _`run_op/task_options[tbtrans_negf]/espacing`: 

        espacing: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[tbtrans_negf]/espacing``

            Energy grid spacing in `unit`. Required.

        .. _`run_op/task_options[tbtrans_negf]/emin`: 

        emin: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[tbtrans_negf]/emin``

            Lower bound of the energy grid in `unit`. Required.

        .. _`run_op/task_options[tbtrans_negf]/emax`: 

        emax: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[tbtrans_negf]/emax``

            Upper bound of the energy grid in `unit`. Required.

        .. _`run_op/task_options[tbtrans_negf]/e_fermi`: 

        e_fermi: 
            | type: ``int`` | ``float``
            | argument path: ``run_op/task_options[tbtrans_negf]/e_fermi``

            Device Fermi level in `unit`. Required.

        .. _`run_op/task_options[tbtrans_negf]/density_options`: 

        density_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/density_options``

            Density integration options (Ozaki or Fiori method).


            Depending on the value of *method*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/density_options/method`: 

            method:
                | type: ``str`` (flag key), default: ``Ozaki``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options/method`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]|_, |code:run_op/task_options[tbtrans_negf]/density_options[Fiori]|_

                Density integration method. 'Ozaki' (default) uses the Ozaki contour with Matsubara-pole expansion; 'Fiori' uses real-axis integration with Gauss quadrature (used with the Fiori/effective-mass approach).

                .. |code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]| replace:: ``Ozaki``
                .. _`code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]`: `run_op/task_options[tbtrans_negf]/density_options[Ozaki]`_
                .. |code:run_op/task_options[tbtrans_negf]/density_options[Fiori]| replace:: ``Fiori``
                .. _`code:run_op/task_options[tbtrans_negf]/density_options[Fiori]`: `run_op/task_options[tbtrans_negf]/density_options[Fiori]`_

            .. |flag:run_op/task_options[tbtrans_negf]/density_options/method| replace:: *method*
            .. _`flag:run_op/task_options[tbtrans_negf]/density_options/method`: `run_op/task_options[tbtrans_negf]/density_options/method`_


            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]`: 

            When |flag:run_op/task_options[tbtrans_negf]/density_options/method|_ is set to ``Ozaki``: 

            Ozaki-contour options.

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/R`: 

            R: 
                | type: ``int`` | ``float``, optional, default: ``1000000.0``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/R``

                Radius of the semicircular contour in `unit`. Default: 1e6.

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/M_cut`: 

            M_cut: 
                | type: ``int``, optional, default: ``30``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/M_cut``

                Number of Ozaki poles kept. Default: 30.

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``10``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/n_gauss``

                Number of Gauss-Legendre points on the equilibrium contour. Default: 10.


            .. _`run_op/task_options[tbtrans_negf]/density_options[Fiori]`: 

            When |flag:run_op/task_options[tbtrans_negf]/density_options/method|_ is set to ``Fiori``: 

            Fiori real-axis integration options.

            .. _`run_op/task_options[tbtrans_negf]/density_options[Fiori]/integrate_way`: 

            integrate_way: 
                | type: ``int`` | ``str``, optional, default: ``direct``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Fiori]/integrate_way``

                Integration strategy for the Fiori method: 'direct' or 'gauss'. Default: 'direct'.

            .. _`run_op/task_options[tbtrans_negf]/density_options[Fiori]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Fiori]/n_gauss``

                Number of Gauss quadrature points along the real-axis integration. Default: 100.

        .. _`run_op/task_options[tbtrans_negf]/eta_lead`: 

        eta_lead: 
            | type: ``int`` | ``float``, optional, default: ``1e-05``
            | argument path: ``run_op/task_options[tbtrans_negf]/eta_lead``

            Small imaginary broadening added to the lead energy. Default: 1e-5.

        .. _`run_op/task_options[tbtrans_negf]/eta_device`: 

        eta_device: 
            | type: ``int`` | ``float``, optional, default: ``0.0``
            | argument path: ``run_op/task_options[tbtrans_negf]/eta_device``

            Small imaginary broadening added to the device energy. Default: 0.

        .. _`run_op/task_options[tbtrans_negf]/out_dos`: 

        out_dos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_dos``

            Whether to write density of states. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_tc`: 

        out_tc: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_tc``

            Whether to write transmission coefficient. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_density`: 

        out_density: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_density``

            Whether to write electron density. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_potential`: 

        out_potential: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_potential``

            Whether to write the self-consistent electrostatic potential. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_current`: 

        out_current: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_current``

            Whether to write self-consistent current. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_current_nscf`: 

        out_current_nscf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_current_nscf``

            Whether to write non-self-consistent current. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_ldos`: 

        out_ldos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_ldos``

            Whether to write local density of states. Default: False.

        .. _`run_op/task_options[tbtrans_negf]/out_lcurrent`: 

        out_lcurrent: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_lcurrent``

            Whether to write local current. Default: False.

    .. _`run_op/structure`: 

    structure: 
        | type: ``NoneType`` | ``str``, optional, default: ``None``
        | argument path: ``run_op/structure``

        Path to the structure file (xyz/vasp/etc.). Overrides the CLI `--structure` flag when set.

    .. _`run_op/pbc`: 

    pbc: 
        | type: ``bool`` | ``list`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/pbc``

        Overall periodic boundary condition of the input structure. Bool applies to all three axes; a list of three bools sets x/y/z individually. Default: None (inferred from the structure file).

    .. _`run_op/use_gui`: 

    use_gui: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``run_op/use_gui``

        Whether to launch the GUI. Default: False.

    .. _`run_op/device`: 

    device: 
        | type: ``NoneType`` | ``str``, optional, default: ``None``
        | argument path: ``run_op/device``

        The torch device used to build the Hamiltonian and run the calculation. Choose 'cpu' or 'cuda[:int]'. Default: None (use the device stored in the model ckpt).

    .. _`run_op/dtype`: 

    dtype: 
        | type: ``NoneType`` | ``str``, optional, default: ``None``
        | argument path: ``run_op/dtype``

        Floating precision used at build time. 'float32' or 'float64'. Default: None (use the dtype stored in the model ckpt).

    .. _`run_op/AtomicData_options`: 

    AtomicData_options: 
        | type: ``NoneType`` | ``dict``, optional, default: ``None``
        | argument path: ``run_op/AtomicData_options``

        .. _`run_op/AtomicData_options/r_max`: 

        r_max: 
            | type: ``int`` | ``float`` | ``dict``
            | argument path: ``run_op/AtomicData_options/r_max``

            Cutoff radius (Angstrom) for bond construction in the TB model. Can be a float or a per-species dict.

        .. _`run_op/AtomicData_options/er_max`: 

        er_max: 
            | type: ``int`` | ``float`` | ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/er_max``

            Cutoff radius for the environment of each site (env-correction models). Default: None.

        .. _`run_op/AtomicData_options/oer_max`: 

        oer_max: 
            | type: ``int`` | ``float`` | ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/oer_max``

            Cutoff radius for onsite environment; only used by nnsk models in strain or NRL mode. Default: None.

