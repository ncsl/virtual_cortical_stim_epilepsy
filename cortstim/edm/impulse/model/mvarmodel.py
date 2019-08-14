import random

import numpy as np
import scipy
from scipy.sparse import linalg, dok_matrix, coo_matrix, csr_matrix

from cortstim.edm.fragility.model.basemodel import BaseWindowModel


# from numba import jit


def test_coo(eegwin):
    # Extract shape of window of eeg to get number of channels and samps
    numchans, numwinsamps = eegwin.shape

    # initialize the step through H matrix
    stepsize = np.arange(0, numchans * (numwinsamps - 1), numchans)
    # stepsize = np.arange(0, numchans * (numwinsamps), numchans)

    # reorder eegwin if necessary
    if eegwin.shape[0] < eegwin.shape[1]:
        eegwin = eegwin.transpose()

    # intialize the data that will be fed into the model
    buff = eegwin[0:numwinsamps - 1, :]

    # get the constant length in each row and column
    rowlen = len(stepsize)
    collen = numchans

    # how long is each slice_array step
    slice_arr_len = rowlen * collen
    slice_arr = np.zeros((rowlen * collen * collen, 2), dtype=np.int)
    buff_mat = np.zeros(slice_arr.shape[0], dtype=np.float)

    for ichan in range(0, numchans):
        # indices to slice through H matrix by
        rowinds = stepsize + (ichan)
        colinds = np.arange((ichan) * numchans, (ichan + 1) * numchans)

        # build the array of the rows,cols in the slice array
        ''' https://stackoverflow.com/questions/43228347/convert-numpy-open-mesh-to-coordinates '''
        m = np.ix_(rowinds, colinds)
        p = np.r_[2:0:-1, 3:len(m) + 1, 0]
        out = np.array(np.meshgrid(*m)).transpose(p).reshape(-1, len(m))
        slice_arr[ichan *
                  slice_arr_len:(ichan + 1) * slice_arr_len, :] = out

        # build up the data vector to insert at each row,col
        buff_mat[ichan *
                 slice_arr_len:(ichan +
                                1) *
                 slice_arr_len] = buff.ravel()

    # 3: compute sparse least squares
    H = coo_matrix((buff_mat, (slice_arr[:, 0], slice_arr[:, 1]))).tocsr()
    return H


def test_csr(eegwin):
    N, W = eegwin.shape

    indices = [x for _ in range(W - 1) for x in range(N ** 2)]
    ptrs = [N * (i) for i in range(N * (W - 1))]
    ptrs.append(len(indices))

    data = []
    for i in range(W - 1):
        vec = eegwin[:, i].squeeze()

        # repeat vector N times
        for i in range(N):
            data.extend(vec)

    Hshape = ((N * (W - 1), N ** 2))
    H = csr_matrix((data, indices, ptrs), shape=Hshape)
    return H


class MvarModel(BaseWindowModel):
    """
    Multivariate-Autoregressive style linear model algorithm used for generating a linear system
    of the style:

    x(t+1) = Ax(t)

    The algorithm takes in a window of data that is NxT, where N is typically the
    number of channels and T is the number of samples for a specific window to generate
    the linear system. It relies on sparse least squares algorithm in scipy and efficiently
    generating the column sparse representation (Csr) matrix in scipy.

    Attributes
    ----------
    stabilize : bool
        Whether or not to stabilize the eigenvalues of the computed A matrix.
    maxeig : float
        The maximum value of the eigenvalues to constrain the A matrix to. Will push all
        eigenvalues in A matrix > maxeig to the maxeig value.
    l2penalty : float
        The l2-norm regularization if 'regularize' is turned on.
    Notes
    -----

    When the size of the data is too large (e.g. N > 180, W > 1000), then right now the construction of the csr
    matrix scales up. With more efficient indexing, we can perhaps decrease this.

    Examples
    --------
    >>> import numpy as np
    >>> from cortstim.edm.fragility.model.mvarmodel import MvarModel
    >>> model_params = {
    ...     'stabilize': False,
    ...     'maxeig': 1.0,
    ...     'l2penalty': 1e-1,
    ...     }
    >>> model = MvarModel(**model_params)
    >>> data = np.random.rand((80,250))
    >>> Amat = model.mvaradjacencymatrix(data)
    >>> print(Amat.shape)
    """

    def __init__(self, l2penalty=0.0,
                 stabilize=False, maxeig=1.0, multitaper=False):
        super(MvarModel, self).__init__()

        self.l2penalty = l2penalty
        self.stabilize = stabilize
        self.maxeig = maxeig
        self.multitaper = multitaper

    def __str__(self):
        return "LTI Window Model {}".format(self.parameters)

    @property
    def parameters(self):
        return {
            'stabilize': self.stabilize,
            'maxeig': self.maxeig,
            'l2penalty': self.l2penalty
        }

    @staticmethod
    def stabilize_matrix(A, eigval):
        """
        Function to stabilize the matrix, A based on its eigenvalues.
        Assumes discrete-time linear system.

        x(t+1) = Ax(t)

        :param A: (np.ndarray) NxN matrix
        :param eigval: (float) the maximum eigenvalue to shift all large evals to
        :return: A, the stabilized matrix
        """
        # check if there are unstable eigenvalues for this matrix
        if np.sum(np.abs(np.linalg.eigvals(A)) > eigval) > 0:
            # perform eigenvalue decomposition
            eigA, V = scipy.linalg.eig(A)

            # get magnitude of these eigenvalues
            abs_eigA = np.abs(eigA)

            # get the indcies of the unstable eigenvalues
            unstable_inds = np.where(abs_eigA > eigval)[0]

            # move all unstable evals to magnitude eigval
            for ind in unstable_inds:
                # extract the unstable eval and magnitude
                unstable_eig = eigA[ind]
                this_abs_eig = abs_eigA[ind]

                # compute scaling factor and rescale eval
                k = eigval / this_abs_eig
                eigA[ind] = k * unstable_eig

            # recompute A
            eigA_mat = np.diag(eigA)
            Aprime = np.dot(V, eigA_mat)
            A = scipy.linalg.lstsq(V.T, Aprime.T)[0].T
        return A

    def mvaradjacencymatrix(self, eegwin):
        """
        Generates adjacency matrix for each window for a certain winsize and stepsize.

        :param eegwin: (np.ndarray) the raw numpy array of data CxT, numchans by numsamps
        :return: adjmat (np.ndarray) the mvar model tensor CxC samples by chan by chan
        """

        # Yield successive n-sized
        # chunks from l.
        def divide_chunks(l, n):
            # looping till length l
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # 1. determine shape of the window of data
        numchans = eegwin.shape[0]

        if self.multitaper:
            numtapers = 50
            thetas = []

            startinds = {}
            while len(startinds.keys()) < numtapers:
                startinds[random.randint(0, eegwin.shape[1] - numchans - 1)] = 1

            for ind in startinds:
                taper_eegwin = eegwin[:, ind:ind + numchans + 1]
                obvector = np.ndarray.flatten(taper_eegwin[:, 1:], order='F')
                # 3. perform sparse least squares and reshape vector -> mat
                H = self._constructsparsemat_viacoo(taper_eegwin)
                # H = self._constructsparsemat_directcsr(eegwin)
                theta = self._compute_lstsq(H, obvector)
                thetas.append(theta)
            thetas = np.array(thetas)
            print(thetas.shape)
            theta = np.mean(thetas, axis=0)
            print(theta.shape)

        else:
            # 2. compute functional connectivity - create observation vector (b) from vectorized data
            obvector = np.ndarray.flatten(eegwin[:, 1:], order='F')
            # 3. perform sparse least squares and reshape vector -> mat
            H = self._constructsparsemat_viacoo(eegwin)
            # H = self._constructsparsemat_directcsr(eegwin)
            theta = self._compute_lstsq(H, obvector)

        adjmat = theta.reshape((numchans, numchans))

        # allow stabilization of the ltv model
        if self.stabilize:
            self.logger.info("Stabilizing matrix! {} to eigenvalue {}".format(
                adjmat.shape, self.maxeig))
            MvarModel.stabilize_matrix(adjmat, self.maxeig)

        # return adjacency matrix to function call
        return adjmat

    # @jit
    def _constructsparsemat_viacoo(self, eegwin):
        """
        Method to compute sparse least squares via scipy's sparse least squares algorithm.

        Uses initialization via coo matrix, which then converts into csr format.

        :param eegwin:
        :param obvector:
        :return:
        """
        # print("Constructing H via coo format first -> csr.")
        # Extract shape of window of eeg to get number of channels and samps
        numchans, numwinsamps = eegwin.shape

        # initialize the step through H matrix
        stepsize = np.arange(0, numchans * (numwinsamps - 1), numchans)
        # stepsize = np.arange(0, numchans * (numwinsamps), numchans)

        # reorder eegwin if necessary
        if eegwin.shape[0] < eegwin.shape[1]:
            eegwin = eegwin.transpose()

        self.logger.info("Constructing a sparse matrix via coo with a "
                         "stepsize of {} and data shape of {}".format(
                             stepsize,
                             eegwin.shape
                         ))

        # intialize the data that will be fed into the model
        buff = eegwin[0:numwinsamps - 1, :].ravel()

        # get the constant length in each row and column
        rowlen = len(stepsize)
        collen = numchans

        # how long is each slice_array step
        slice_arr_len = rowlen * collen
        slice_arr = np.zeros((rowlen * collen * collen, 2), dtype=np.int)
        buff_mat = np.zeros(slice_arr.shape[0], dtype=np.float)

        for ichan in range(0, numchans):
            # indices to slice through H matrix by
            rowinds = stepsize + (ichan)
            colinds = np.arange((ichan) * numchans, (ichan + 1) * numchans)

            # build the array of the rows,cols in the slice array
            ''' https://stackoverflow.com/questions/43228347/convert-numpy-open-mesh-to-coordinates '''
            m = np.ix_(rowinds, colinds)
            p = np.r_[2:0:-1, 3:len(m) + 1, 0]
            out = np.array(np.meshgrid(*m)).transpose(p).reshape(-1, len(m))
            slice_arr[ichan *
                      slice_arr_len:(ichan + 1) * slice_arr_len, :] = out

            # build up the data vector to insert at each row,col
            buff_mat[ichan * slice_arr_len:(ichan + 1) * slice_arr_len] = buff

        N = numchans
        W = numwinsamps
        Hshape = ((N * (W - 1), N ** 2))

        self.logger.info("Analyzing {numchans} contacts to create a H matrix ({Hshape})".format(
            numchans=numchans,
            Hshape=Hshape,
        ))

        # 3: compute sparse least squares
        H = coo_matrix(
            (buff_mat, (slice_arr[:, 0], slice_arr[:, 1])), shape=Hshape, dtype=np.float64).tocsr()
        return H

    def _constructsparsemat_directcsr(self, eegwin):
        """
        Method to compute sparse least squares via scipy's sparse least squares algorithm.

        Uses direct initialization into csr format, which should theoretically provide significant speedups
        for matrices that are larger (i.e. more channels and more time windows).

        :param eegwin: (np.ndarray) the raw numpy array of data CxW, numchans by numsamps
        :return: H (scipy.csr_matrix) sparse H matrix that is (N*W x N*N)
        """
        print("Constructing H directly to csr format.", flush=True)
        # Extract shape of window of eeg to get number of channels and samps
        eegwin = eegwin[:, :-1]
        N, W = eegwin.shape

        ptrs = N * np.arange(W * N + 1, dtype='int')
        indices = np.tile(np.arange(N * N, dtype='int'), W)
        data = np.tile(eegwin, N).flatten()

        Hshape = ((N * (W), N ** 2))

        # print(len(data), len(indices), len(ptrs))
        # 3: compute sparse least squares
        H = csr_matrix((data, indices, ptrs), shape=Hshape, dtype=np.float64)
        return H

    def _constructsparsemat_viadok(self, eegwin):
        '''
        # Least Squares wrapper function that constructs via dok
        # FUNCTION:
        #   y = computelstsq(eegWin, obvector)
        #
        # INPUT ARGS: (defaults shown):
        #   eegWin = dat;           # CxT matrix with C chans and T samps, the A in Ax=b
        #   obvector = [58 62];     # Tx1 vector, the b in Ax=b
        # OUTPUT ARGS::
        #   theta = vector of weights in the x_ij in MVAR model
        # Extract shape of window of eeg to get number of channels and samps
        '''
        print("Constructing sparse matrix via dok.")

        numchans, numwinsamps = eegwin.shape

        # initialize H matrix as array filled with zeros
        H = dok_matrix((numchans * (numwinsamps - 1), numchans ** 2))

        # 1: fill all columns in matrix H
        eegwin = eegwin.transpose()
        stepsize = np.arange(0, H.shape[0], numchans)
        H[stepsize, 0:numchans] = eegwin[0:numwinsamps - 1, :]

        buff = eegwin[0:numwinsamps - 1, :]
        for ichan in range(1, numchans):
            # indices to slice through H matrix by
            rowinds = stepsize + (ichan)
            colinds = np.arange((ichan) * numchans, (ichan + 1) * numchans)
            # slice through H mat and build it
            H[np.ix_(rowinds, colinds)] = buff

        return H.tocsr()

    # @jit
    def _compute_lstsq(self, H, obvector):
        """
        Wrapper function to compute least squares.

        Allows user to penalize it with different regularizations if necessary.

        :param H: (scipy.csr_matrix) sparse H matrix that is (N*W x N*N)
        :param obvector: (np.ndarray) observation vector in least-squares
        :return: theta (np.ndarray) in Ax = y => solve for x
        """
        self.logger.info("Performing least-squares estimation of A matrix "
                         "with l2-regularization set to {}".format(
                             self.l2penalty,
                         ))
        # perform least squares scipy optimization
        theta = linalg.lsqr(H, obvector, damp=self.l2penalty)[0]
        return theta
