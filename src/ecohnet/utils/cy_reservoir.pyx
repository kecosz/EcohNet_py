
cimport cython
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.math cimport fabs, tanh

import numpy as np

from numpy cimport float64_t, uint32_t
from numpy.random cimport bitgen_t
from numpy.random.bit_generator cimport BitGenerator
from numpy.random.c_distributions cimport (random_standard_uniform_fill,
                                           random_uniform)

from numpy.random import PCG64

np.import_array()

from .constant import Const
from .math import RMSE


cpdef float calc_bagg(double[:, ::1] source, double[::1] target):
    nin, nx, win, ww, ilamb, delta = initializeNN(source.shape[1])
    tgxe = Reservoirepd(source, target, nin, nx, win, ww, ilamb, delta, Const.p4)
    return RMSE(target, tgxe)


cpdef np.ndarray init_reservoirepd(double[:, ::1] source, double[::1] target):
    nin, nx, win, ww, ilamb, delta = initializeNN(source.shape[1])
    tgxe = Reservoirepd(source, target, nin, nx, win, ww, ilamb, delta, Const.p4)
    return tgxe


def calc_baggs(sources: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([calc_bagg(source, target) for source in sources])


def calc_bagg_and_tgtest(input, target):
    """RC_prd の補助関数"""
    tgtest = Reservoirep(input, target, initializeNN(len(input[0])))
    return RMSE(target, tgtest), tgtest


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray Reservoirepd(
    double[:, ::1] inn,
    double[:] out,
    int nin,
    int nx,
    double[:, ::1] win,
    double[:, ::1] ww,
    double ilambda,
    double delta,
    int rls_n,
    int seed = 42,
):
    """Predict multivariate time series with ESN-RLS
    (echo state network updated by recursive least square method).
    The difference from the `Reservoiep` function is the addition of a dropout layer.

    Parameters
    ----------
    inn : np.ndarray
        Input time series, 2-D array of the shape (length of time series, number of variables).
    out : np.ndarray
        Target variable, 1-D array of the length of time series.
        output[t] is predicted from input[:t+1], including input[t] (target is already delayed).
    parameters : tuple[Any]
        input: np.ndarray,
        output: np.ndarray,
        nin: int,
        nx: int,
        win: np.ndarray,
        ww: np.ndarray,
        ilambda: float,
        delta: float,

    Returns
    -------
    np.ndarray :
        Predicted target variable, 1-D array of the length of the time series.
    """
    cdef BitGenerator rng = PCG64(seed)
    cdef int nt = inn.shape[0]
    #cdef int rls_n = Const.p4
    cdef double[::1] yprds = np.zeros(nt)

    cdef double[::1] wou = np.zeros(nx)
    cdef double[::1] xi = np.zeros(nx)
    cdef double[:, ::1] pn = np.eye(nx) / delta
    cdef double[::1] l1 = np.empty(nx)
    cdef double[::1] gn = np.empty(nx)

    cdef double vn, ouprd, tmp
    cdef Py_ssize_t wi, wj, i, nj, rls_i
    cdef uint32_t drop

    for i in range(nt):
        # l1 = np.tanh(win @ inx * random.integer(0,2) + ww @ xi)
        drop = bounded_uint(rng, 0, 1)
        for wi in range(nx):
            l1[wi] = 0
            if drop == 1:
                for nj in range(nin):
                    l1[wi] += win[wi, nj] * inn[i, nj]
            for wj in range(nx):
                l1[wi] += ww[wi, wj] * xi[wj]
            l1[wi] = tanh(l1[wi])

        # yprds[i] = wou @ l1
        for wi in range(nx):
            yprds[i] += wou[wi] * l1[wi]

        for rls_i in range(rls_n):
            # ouprd = wou @ l1
            ouprd = 0
            for wi in range(nx):
                ouprd += wou[wi] * l1[wi]
            vn = out[i] - ouprd
            # gn = ilambda * (pn @ xi) / (1 + ilambda * (xi @ pn) @ xi)
            tmp = 0
            for wi in range(nx):
                for wj in range(nx):
                    tmp += xi[wi] * pn[wi, wj] * xi[wj]
            tmp = 1 + ilambda * tmp
            for wi in range(nx):
                gn[wi] = 0
                for wj in range(nx):
                    gn[wi] += pn[wi, wj] * xi[wj]
                gn[wi] = gn[wi] * ilambda / tmp
            # pn = ilambda * (pn - (gn @ xi) * pn)
            tmp = 0
            for wi in range(nx):
                tmp += gn[wi] * xi[wi]
            for wi in range(nx):
                for wj in range(nx):
                    pn[wi, wj] = ilambda * (pn[wi, wj] - tmp * pn[wi, wj])
            # state update
            xi[:] = l1[:]
            # wou = wou + gn * vn
            for wi in range(nx):
                wou[wi] = wou[wi] + gn[wi] * vn
    return np.asarray(yprds)


cpdef np.ndarray Reservoirep(
    double[:, ::1] inn,
    double[:] out,
    parameters,
):
    """Predict multivariate time series with ESN-RLS"""
    return comp(inn, out, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])


cdef np.ndarray comp(
    double[:, ::1] inn,
    double[:] out,
    int nin,
    int nx,
    double[:, ::1] win,
    double[:, ::1] ww,
    double ilambda,
    double delta,
):
    """Auxiliary functions for the `Reservoirep` function."""
    cdef int nt = inn.shape[0]
    cdef int rls_n = Const.p4
    cdef double[::1] yprds = np.zeros(nt)

    cdef double[::1] wou = np.zeros(nx)
    cdef double[::1] xi = np.zeros(nx)
    cdef double[:, ::1] pn = np.eye(nx) / delta
    cdef double[::1] l1 = np.empty(nx)
    cdef double[::1] gn = np.empty(nx)

    cdef double vn, ouprd, tmp
    cdef Py_ssize_t wi, wj, i, nj, rls_i

    for i in range(nt):
        # l1 = np.tanh(win @ inx + ww @ xi)
        for wi in range(nx):
            l1[wi] = 0
            for nj in range(nin):
                l1[wi] += win[wi, nj] * inn[i, nj]
            for wj in range(nx):
                l1[wi] += ww[wi, wj] * xi[wj]
            l1[wi] = tanh(l1[wi])

        # yprds[i] = wou @ l1
        for wi in range(nx):
            yprds[i] += wou[wi] * l1[wi]

        for rls_i in range(rls_n):
            # ouprd = wou @ l1
            ouprd = 0
            for wi in range(nx):
                ouprd += wou[wi] * l1[wi]
            vn = out[i] - ouprd
            # gn = ilambda * (pn @ xi) / (1 + ilambda * (xi @ pn) @ xi)
            tmp = 0
            for wi in range(nx):
                for wj in range(nx):
                    tmp += xi[wi] * pn[wi, wj] * xi[wj]
            tmp = 1 + ilambda * tmp
            for wi in range(nx):
                gn[wi] = 0
                for wj in range(nx):
                    gn[wi] += pn[wi, wj] * xi[wj]
                gn[wi] = gn[wi] * ilambda / tmp
            # pn = ilambda * (pn - (gn @ xi) * pn)
            tmp = 0
            for wi in range(nx):
                tmp += gn[wi] * xi[wi]
            for wi in range(nx):
                for wj in range(nx):
                    pn[wi, wj] = ilambda * (pn[wi, wj] - tmp * pn[wi, wj])
            # state update
            xi[:] = l1[:]
            # wou = wou + gn * vn
            for wi in range(nx):
                wou[wi] = wou[wi] + gn[wi] * vn
    return np.asarray(yprds)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple initializeNN(int nin, int seed = 42):
    """Return a new instance of ESN(echo state network).

    Parameter
    ---------
    nin : int
        Number of input variables.

    Returns
    -------
    nin : int
        Number of input variables.
    nx : int
        Number of recurrent nodes.
    win : np.ndarray.
        Input weight matrix, 1-D array of the shape (nx, nin).
    ww : np.ndarray
        Reservoir weight matrix, 2-D array of the shape (nx, nx).
    ilambda : float
        Inverse of forgetting rate in RLS.
    delta : float
        Regularize coefficient in RLS.
    """
    cdef int nx = Const.q1
    cdef double radius = Const.p1
    cdef double lamb = Const.p2
    cdef double delta = Const.p3

    cdef double gamma, i_lambda, scwin, conn, rho0
    cdef np.ndarray[float64_t, ndim=2] win

    cdef BitGenerator rng = PCG64()


    # make win
    scwin = 0.1
    win = uniforms(rng, nx*nin, -1, 1).reshape((nx, nin)) * scwin
    # make ww, weight of resorvoir network
    conn = 0.1
    cdef double[:, ::1] ww0 = np.empty((nx, nx))
    cdef double[::1] sum0 = np.empty(nx)
    cdef double[::1] sum1 = np.empty(nx)

    while True:
        # make ww0
        for i in range(nx):
            for j in range(nx):
                if uniform(rng, 0, 1) < conn:
                    ww0[i, j] = uniform(rng, -1, 1)
                else:
                    ww0[i, j] = 0
        # crit = sum(np.abs((sum(ww0) * sum(ww0.T))) < 10e-8)
        # if crit == 0: break
        for i in range(nx):
            sum0[i] = 0
            sum1[i] = 0
            for j in range(nx):
                sum0[i] += ww0[i, j]
                sum1[i] += ww0[i, j]
        for i in range(nx):
            if fabs(sum0[i] * sum1[i]) < 1e-8:
                break
        else:
            break

    ww0_array = np.asarray(ww0)
    rho0 = max(abs(np.linalg.eigvals(ww0_array)))
    gamma = radius * (1 / rho0)
    ww = gamma * ww0_array
    # make others
    ilamb = 1 / lamb
    return nin, nx, win, ww, ilamb, delta


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint32_t bounded_uint(BitGenerator bit_generator, uint32_t lb, uint32_t ub):
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"

    # rng = <bitgen_t *> PyCapsule_GetPointer(bit_generator.capsule, capsule_name)
    rng = <bitgen_t *> PyCapsule_GetPointer(bit_generator.capsule, capsule_name)

    cdef uint32_t mask, delta, val
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    val = rng.next_uint32(rng.state) & mask
    while val > delta:
        val = rng.next_uint32(rng.state) & mask

    return lb + val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint32_t bounded_uints(BitGenerator bit_generator, int n, uint32_t lb, uint32_t ub):
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"

    rng = <bitgen_t *> PyCapsule_GetPointer(bit_generator.capsule, capsule_name)

    cdef uint32_t mask, delta, val
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    cdef uint32_t[::1] uints = np.zeros(n, dtype=np.uint32)
    with bit_generator.lock, nogil:  # type: ignore
        for i in range(n):
            val = rng.next_uint32(rng.state) & mask
            while val > delta:
                val = rng.next_uint32(rng.state) & mask
            uints[i] = lb + val
    return np.asarray(uints)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double uniform(BitGenerator bit_generator, double lower, double higher):
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"

    rng = <bitgen_t*> PyCapsule_GetPointer(bit_generator.capsule, capsule_name)
    with bit_generator.lock, nogil:  # type: ignore
        return random_uniform(rng, lower, higher-lower)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray uniforms(BitGenerator bit_generator, int n, double lower, double higher):
    """
    Create an array of `n` uniformly distributed doubles.
    """
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef np.ndarray[float64_t, ndim=1] randoms

    rng = <bitgen_t*> PyCapsule_GetPointer(bit_generator.capsule, capsule_name)
    randoms = np.empty(n, dtype=np.float64)
    with bit_generator.lock, nogil:  # type: ignore
        random_standard_uniform_fill(rng, n, <double *>np.PyArray_DATA(randoms))
    randoms = lower + randoms * (higher - lower)
    return randoms

