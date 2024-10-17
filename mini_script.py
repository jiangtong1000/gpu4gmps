"""Modules for testing the performance of GPU GMPS acceleration."""

import copy
from numba import njit
import numpy
import numpy as np
import scipy
import time

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

    # bond_dim = [1] * len(bond_dim)
    mps = []
    for i in range(nsite):
        random_mat = scipy.stats.unitary_group.rvs(bond_dim[i] * 2)[:, :bond_dim[i + 1]]
        mps.append(random_mat.reshape(bond_dim[i], 2, bond_dim[i + 1]))
    return mps

@njit()
def fake_givens_gate(n_gate):
    """
    Generate a list of random Givens gates.
    """
    gates = numpy.zeros((n_gate, 4, 4))
    for i in range(n_gate):
        sin = -numpy.random.rand()
        cos = numpy.sqrt(1 - sin**2)
        gates[i][1:3, 1:3] = numpy.array([[cos, sin], [-sin, cos]])
        gates[i][0, 0] = 1
        gates[i][3, 3] = 1
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
    total_layers = max([nsites - 1 - i_site + sites_depth[i_site] - 1 for i_site in range(nsites - 1)])
    layer_gates = {i: None for i in range(total_layers)}
    layer_target_sites = {i: [] for i in range(total_layers)}

    for i in range(nsites - 2, -1, -1):
        start_layer = nsites - 2 - i
        layer_target_sites[start_layer].append([i, i + 1])
        num_of_gates = sites_depth[i]
        for j in range(1, num_of_gates):
            start_layer += 1
            layer_target_sites[start_layer].append([i + j, i + j + 1])
    for i_layer in range(total_layers):
        num_of_gates = len(layer_target_sites[i_layer])
        layer_gates[i_layer] = fake_givens_gate(num_of_gates)
    return layer_gates, layer_target_sites


def fake_big_mps(nsite, bond_dim, num_of_mps, max_depth_per_site, mps=None):
    """
    Generate a list of MPS, each MPS has the same structure.
    """
    if mps is None:
        mps = fake_mps(nsite, bond_dim)
    big_mps = [copy.deepcopy(imps) for _ in range(num_of_mps) for imps in mps]
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
def gate_contraction_split(
        local_op, mps_site1, mps_site2):


    diml1, dima1, dimr1 = mps_site1.shape
    mps_site1 = mps_site1.reshape(diml1 * dima1, dimr1)
    diml2, dima2, dimr2 = mps_site2.shape
    mps_site2 = mps_site2.reshape(diml2, dima2 * dimr2)
    mps_site12 = mps_site1.dot(mps_site2)
    mps_site12 = mps_site12.reshape(diml1, 4, dimr2)

    op_mps = np.zeros((diml1, 2, 2, dimr2), dtype=np.complex128)
    local_op = local_op.reshape(2, 2, 4)

    args = [[0, 1, 1], [0, 1, 2], [1, 0, 1], [1, 0, 2]]
    for a, b, e in args:
        for c in range(diml1):
            for d in range(dimr2):
                op_mps[c, a, b, d] += local_op[a, b, e] * mps_site12[c, e, d]

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
            # op_mps = numpy.einsum("abcd, ecf, fdg->eabg", i_gate, mps[i], mps[j])

            if not mps[i].flags['C_CONTIGUOUS']:
                mps[i] = mps[i].copy(order="C")
            if not mps[j].flags['C_CONTIGUOUS']:
                mps[j] = mps[j].copy(order="C")
            op_mps = gate_contraction_split(i_gate, mps[i], mps[j])

            diml = mps[i].shape[0]
            dimr = mps[j].shape[2]
            op_mps = op_mps.reshape(diml * 2, dimr * 2)

            # q, r = scipy.linalg.qr(op_mps, mode="economic")
            # imps_1 = q.reshape(q.shape[0]//2, 2, -1)
            # imps_2 = r.reshape(-1, 2, r.shape[1]//2)

            u, s, vdag = scipy.linalg.svd(
                op_mps, full_matrices=False, lapack_driver="gesvd"
            )
            num_of_states = min(bond_dim, len(s))
            u = u[:, :num_of_states] @ numpy.diag(s[:num_of_states])
            imps_1 = u.reshape(-1, 2, num_of_states)
            imps_2 = (vdag[:num_of_states, :]).reshape(num_of_states, 2, -1)

            mps[i] = copy.deepcopy(imps_1)
            mps[j] = copy.deepcopy(imps_2)
    return mps


def gpu_contract(mps, layer_gates, layer_target_sites, max_bond_dim, unit_n_site):
    """
    Use cupy einsum to contract the stacked MPS, layer by layer.
    """
    total_layers = len(layer_gates)
    # Preallocate memory once for the padding template outside the loop
    mps_padded = numpy.zeros((max_bond_dim, 2, max_bond_dim), dtype=numpy.complex128)

    for i in range(len(mps)):
        diml, dima, dimr = mps[i].shape
        mps_padded.fill(0)
        mps_padded = numpy.zeros((max_bond_dim, 2, max_bond_dim), dtype=numpy.complex128)
        mps_padded[:diml, :, :dimr] = mps[i]
        mps[i] = np.copy(mps_padded)

    for i_layer in range(total_layers):
        # make regular shape to be (bond_dim, 2, bond_dim)
        # padding zeros if the bond_dim is smaller than the largest bond_dim
        mps_i_merge = []
        mps_j_merge = []
        gates = []

        for idx, (i, j) in enumerate(layer_target_sites[i_layer]):
            mps_i_merge.append(mps[i])
            mps_j_merge.append(mps[j])
            gates.append(layer_gates[i_layer][idx])
        mps_i_merge = xp.array(mps_i_merge)  # shape (n_gate, bond_dim, 2, bond_dim)
        mps_j_merge = xp.array(mps_j_merge)  # shape (n_gate, bond_dim, 2, bond_dim)
        gates = xp.array(gates)  # shape (n_gate, 2, 2, 2, 2)
        n_gate = len(gates)

        op_mps = xp.einsum(
                "nabcd, necf, nfdg->neagb", gates, mps_i_merge, mps_j_merge, optimize=True
            )
        op_mps = op_mps.reshape(n_gate, max_bond_dim * 2, max_bond_dim * 2)
        u, s, vdag = xp.linalg.svd(op_mps, full_matrices=False)
        num_of_states = min(max_bond_dim, s.shape[1])
        u = u[:, :, :num_of_states]
        s = s[:, :num_of_states]
        us = xp.einsum("nab,nb->nab", u, s, optimize=True)
        mps_i = us.reshape(n_gate, max_bond_dim, 2, num_of_states)
        mps_j = vdag[:, :num_of_states, :].reshape(n_gate, num_of_states, -1, 2)
        mps_j = mps_j.transpose(0, 1, 3, 2)

        for idx, (i, j) in enumerate(layer_target_sites[i_layer]):
            diml, dimr = mps_i[idx].shape[0], mps_i[idx].shape[2]
            mps_padded.fill(0)
            mps_padded[:diml, :, :dimr] = mps_i[idx]
            mps[i] = xp.copy(mps_padded)

            diml, dimr = mps_j[idx].shape[0], mps_j[idx].shape[2]
            mps_padded.fill(0)
            mps_padded[:diml, :, :dimr] = mps_j[idx]
            mps[j] = xp.copy(mps_padded)
    return mps

def cast_gpu_mps_back(mps, nsite, num_of_mps):
    """
    Cast the GPU MPS back to CPU MPS.
    """
    assert len(mps) == nsite * num_of_mps
    for i in range(len(mps)):
        if i % nsite == 0:
            assert numpy.allclose(mps[i][1:], 0)
            mps[i] = mps[i][:1]
        else:
            match_bond_dim = min(mps[i].shape[0], mps[i-1].shape[2])
            assert numpy.allclose(mps[i][match_bond_dim:], 0)
            mps[i] = mps[i][:match_bond_dim]
            assert numpy.allclose(mps[i-1][:, :, match_bond_dim:], 0)
            mps[i-1] = mps[i-1][:, :, :match_bond_dim]
        if (i+1) % nsite == 0:
            mps[i] = mps[i][:, :, :1]
    return mps

# @njit(nopython=True, fastmath=True)
def mps_overlap(mps1, mps2):
    # assert len(mps1) == len(mps2)
    res = numpy.ones((1, 1), dtype=numpy.complex128)
    for i in range(len(mps1)-1, -1, -1):
        l1, m1, r1 = mps1[i].conj().shape
        mps1i_conj = mps1[i].conj().reshape(l1*m1, r1)
        # mps1i_conj = mps1i_conj.astype(numpy.complex128)
        tmp = mps1i_conj.dot(res)
        l2, m2, r2 = mps2[i].shape
        mps2i = mps2[i].reshape(l2*m2, r2).T
        # mps2i = mps2i.astype(numpy.complex128)
        tmp = tmp.dot(mps2i)
        tmp = tmp.reshape(l1, m1, l2, m2)
        res = tmp[:, 0, :, 0] + tmp[:, 1, :, 1]
    return res[0][0]

if __name__ == "__main__":
    nsite = 10
    num_of_mps = 2
    max_depth_per_site = 4
    max_bond_dim = 200
    mps_origin, layer_gates, layer_target_sites = fake_big_mps(
        nsite, max_bond_dim, num_of_mps, max_depth_per_site
    )
    print('total layer is', len(layer_gates))

    print("Start CPU contraction")
    cpu_time = []
    for i in range(3):
        mps = copy.deepcopy(mps_origin)
        start = time.time()
        mps_cpu = cpu_contract(mps, layer_gates, layer_target_sites, max_bond_dim)
        cpu_time.append(time.time()-start)
    print('time cost:', numpy.mean(cpu_time[1:]))
    print('mps.length:', len(mps_cpu))
    print('mps.shape', [i.shape for i in mps_cpu])
    print("Start GPU contraction")
    gpu_time = []
    for i in range(1):
        start = time.time()
        mps = copy.deepcopy(mps_origin)
        mps_gpu = gpu_contract(mps, layer_gates, layer_target_sites, max_bond_dim, nsite)
        mps_gpu = cast_gpu_mps_back(mps_gpu, nsite, num_of_mps)
        gpu_time.append(time.time()-start)
    print('time cost:', numpy.mean(gpu_time[1:]))
    print('mps.length:', len(mps_gpu))
    print('mps.shape', [i.shape for i in mps_gpu])
    print("Finished")
    norm_cpu = numpy.sqrt(mps_overlap(mps_cpu, mps_cpu))
    norm_gpu = numpy.sqrt(mps_overlap(mps_gpu, mps_gpu))
    overlap = mps_overlap(mps_cpu, mps_gpu) / (norm_cpu * norm_gpu)
    print(overlap, numpy.abs(overlap))
