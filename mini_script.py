"""Modules for testing the performance of GPU GMPS acceleration."""

import copy
from numba import njit
import numpy
import scipy

xp = numpy
BACKEND = "numpy"
try:
    import cupy

    xp = cupy
    BACKEND = "cupy"
except ImportError:
    pass


def fake_mps(nsite, max_d):
    """
    Generate a random MPS.

    Parameters:
    - nsite (int): Number of sites in the MPS (must be even).
    - max_d (int): Maximum bond dimension.

    Returns:
    - mps (list of numpy.ndarray): The generated MPS tensors.
    """
    assert nsite % 2 == 0
    bond_dim = [
        min(max_d, 2 ** (i + 1), 2 ** (nsite - i - 1)) for i in range(nsite - 1)
    ]
    bond_dim = [1] + bond_dim + [1]
    mps = []
    for i in range(nsite):
        mps.append(
            numpy.random.random((bond_dim[i], 2, bond_dim[i + 1]))
            + 1j * numpy.random.random((bond_dim[i], 2, bond_dim[i + 1]))
        )
    return mps


def fake_givens_gate(n_gate):
    """
    Generate a list of random Givens gates.
    """
    gates = numpy.zeros((n_gate, 4, 4))
    for i in range(n_gate):
        sin = -numpy.random.rand()
        cos = numpy.sqrt(1 - sin**2)
        gates[i][1:3, 1:3] = numpy.array([[cos, sin], [-sin, cos]])
    gates = gates.reshape(-1, 2, 2, 2, 2)  # pylint: disable=E1121
    return gates


def generate_gate_map(nsites, max_depth):
    """
    Generate a map of gates and target sites.
    """
    sites_depth = []
    for i in range(nsites - 1):
        max_num_of_gates = min(nsites - 1 - i, max_depth)
        num_of_gates = numpy.random.choice(range(1, max_num_of_gates + 1))
        sites_depth.append(num_of_gates)
    total_layers = nsites - 1 + sites_depth[0] - 1
    layer_gates = {i: [] for i in range(total_layers)}
    layer_target_sites = {i: [] for i in range(total_layers)}

    for i in range(nsites - 2, -1, -1):
        start_layer = nsites - 2 - i
        max_num_of_gates = min(nsites - 1 - i, max_depth)
        num_of_gates = numpy.random.choice(range(1, max_num_of_gates + 1))
        layer_gates[start_layer].append(fake_givens_gate(num_of_gates))
        layer_target_sites[start_layer].append([i, i + 1])
        for j in range(1, num_of_gates):
            layer_target_sites[start_layer].append([i + j, i + j + 1])
        assert len(layer_target_sites[start_layer]) == num_of_gates
    return layer_gates, layer_target_sites


def fake_big_mps(nsite, bond_dim, num_of_mps, max_depth_per_site, mps=None):
    """
    Generate a list of MPS, each MPS has the same structure.
    """
    if mps is None:
        mps = fake_mps(nsite, bond_dim)
    big_mps = [copy.deepcopy(imps) for _ in range(num_of_mps) for imps in mps]
    print(len(big_mps))
    layer_gates, layer_target_sites = generate_gate_map(nsite, max_depth_per_site)
    total_layers = len(layer_gates)
    big_layer_gates = {i: [] for i in range(total_layers)}
    for i in range(total_layers):
        big_layer_gates[i] = [
            copy.deepcopy(gate) for _ in range(num_of_mps) for gate in layer_gates[i]
        ]
    big_layer_target_sites = {i: [] for i in range(total_layers)}
    for i in range(total_layers):
        big_layer_target_sites[i] = layer_target_sites[i].copy()
        for i_mps in range(1, num_of_mps):
            shifted_sites = numpy.array(layer_target_sites[i]) + i_mps * nsite
            shifted_sites = shifted_sites.tolist()
            big_layer_target_sites[i].extend(shifted_sites)
    return big_mps, big_layer_gates, big_layer_target_sites


@njit(nopython=True, fastmath=True)
def gate_contraction_split(local_op, mps_site1, mps_site2):
    """
    Using numba jit to contract the local operator with two MPS sites.
    """
    diml1, dima1, dimr1 = mps_site1.shape
    mps_site1 = mps_site1.reshape(diml1 * dima1, dimr1)
    diml2, dima2, dimr2 = mps_site2.shape
    mps_site2 = mps_site2.reshape(diml2, dima2 * dimr2)
    mps_site12 = mps_site1.dot(mps_site2)
    mps_site12 = mps_site12.reshape(diml1, 4, dimr2)

    op_mps = numpy.zeros((diml1, 2, 2, dimr2), dtype=numpy.complex128)
    local_op = local_op.reshape(2, 2, 4)

    args = [[0, 1, 1], [0, 1, 2], [1, 0, 1], [1, 0, 2]]
    for a, b, e in args:
        op_mps[:, a, b, :] += local_op[a, b, e] * mps_site12[:, e, :]

    op_mps[:, 0, 0, :] = mps_site12[:, 0, :]
    op_mps[:, 1, 1, :] = mps_site12[:, 3, :]
    return op_mps


def cpu_contract(mps, layer_gates, layer_target_sites, bond_dim):
    """
    Use numpy einsum to contract the stacked MPS, layer by layer.
    For the purpose of testing, TODO:
    We can only contract 1 small MPS, and times the time by the number of MPS.
    """
    total_layers = len(layer_gates)
    for i_layer in range(total_layers):
        for idx, i_gate in enumerate(layer_gates[i_layer]):
            i, j = layer_target_sites[i_layer][idx]
            op_mps = gate_contraction_split(i_gate, mps[i], mps[j])
            diml = mps[i].shape[0]
            dimr = mps[j].shape[2]
            op_mps = op_mps.reshape(diml * 2, dimr * 2)
            u, s, vdag = scipy.linalg.svd(
                op_mps, full_matrices=False, lapack_driver="gesvd"
            )
            num_of_states = min(bond_dim, len(s))
            imps_1 = u[:, :num_of_states].reshape(-1, 2, num_of_states)
            imps_2 = (
                numpy.diag(s[:num_of_states])
                .dot(vdag[:num_of_states, :])
                .reshape(num_of_states, 2, -1)
            )
            mps[i] = imps_1
            mps[j] = imps_2
    return mps


def gpu_contract(mps, layer_gates, layer_target_sites, bond_dim):
    """
    Use cupy einsum to contract the stacked MPS, layer by layer.
    """
    total_layers = len(layer_gates)
    for i_layer in range(total_layers):
        n_gate = len(layer_gates[i_layer])
        largest_bond_dim_left = max(
            [mps[i[0]].shape[0] for i in layer_target_sites[i_layer]]
        )
        largest_bond_dim_right = max(
            [mps[i[1]].shape[2] for i in layer_target_sites[i_layer]]
        )
        bond_dim = min(bond_dim, max([largest_bond_dim_left, largest_bond_dim_right]))
        # make regular shape to be (bond_dim, 2, bond_dim)
        # padding zeros if the bond_dim is smaller than the largest bond_dim
        mps_i_merge = []
        mps_j_merge = []
        for i, j in layer_target_sites[i_layer]:
            diml, dima, dimr = mps[i].shape
            mps_padded = numpy.zeros((bond_dim, dima, bond_dim), dtype=numpy.complex128)
            mps_padded[:diml, :, :dimr] = mps[i]
            mps[i] = mps_padded.copy()
            diml, dima, dimr = mps[j].shape
            mps_padded = numpy.zeros((bond_dim, dima, bond_dim), dtype=numpy.complex128)
            mps_padded[:diml, :, :dimr] = mps[j]
            mps[j] = mps_padded.copy()
            mps_i_merge.append(mps[i])
            mps_j_merge.append(mps[j])
        mps_i_merge = xp.array(mps_i_merge)  # shape (n_gate, bond_dim, 2, bond_dim)
        mps_j_merge = xp.array(mps_j_merge)  # shape (n_gate, bond_dim, 2, bond_dim)
        gates = xp.array(layer_gates[i_layer])  # shape (n_gate, 2, 2, 2, 2)
        op_mps = xp.einsum(
            "nabcd, necf, nfdg->neabg", gates, mps_i_merge, mps_j_merge, optimize=True
        )
        op_mps = op_mps.reshape(n_gate, bond_dim * 2, bond_dim * 2)
        u, s, vdag = xp.linalg.svd(op_mps, full_matrices=False)
        mps_i = xp.einsum(
            "nab, nc->nacb", u[:, :, :bond_dim], xp.diag(s), optimize=True
        )
        mps_i = mps_i.reshape(n_gate, bond_dim, 2, bond_dim)
        mps_j = vdag[:, :bond_dim, :].reshape(n_gate, bond_dim, 2, bond_dim)
        for idx, (i, j) in enumerate(layer_target_sites[i_layer]):
            mps[i] = mps_i[idx].get()
            mps[j] = mps_j[idx].get()
