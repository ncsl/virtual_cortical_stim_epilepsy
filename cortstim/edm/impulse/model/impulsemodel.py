import numpy as np
import numpy.matlib as nmb
import scipy.signal
# from tqdm import trange


class NormalizedModel(object):
    @staticmethod
    def compute_fragilitymetric(minnormpertmat):
        """
        Function to perform column normalization of a matrix that computes:

        a_ij = (max_in_col(a) - a_ij) / max_in_col(a)

        :param minnormpertmat: (np.ndarray) a NxW matrix
        :return: (np.ndarray) normalized NxW matrix
        """
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        fragilitymat = np.zeros((N, T))
        for icol in range(T):
            fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) / \
                                    np.max(minnormpertmat[:, icol])
        return fragilitymat

    @staticmethod
    def compute_minmaxfragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        minmax_fragilitymat = np.zeros((N, T))

        # get the min/max for each column in matrix
        minacrosstime = np.min(minnormpertmat, axis=0)
        maxacrosstime = np.max(minnormpertmat, axis=0)

        # normalized data with minmax scaling
        minmax_fragilitymat = -1 * np.true_divide((minnormpertmat - nmb.repmat(maxacrosstime, N, 1)),
                                                  nmb.repmat(maxacrosstime - minacrosstime, N, 1))
        return minmax_fragilitymat

    @staticmethod
    def compute_znormalized_fragilitymetric(minnormpertmat):
        # get mean, std
        avg_contacts = np.mean(minnormpertmat, keepdims=True, axis=1)
        std_contacts = np.std(minnormpertmat, keepdims=True, axis=1)

        # normalized data with minmax scaling
        return (minnormpertmat - avg_contacts) / std_contacts


class ImpulseResponse(object):
    def __init__(self, index):
        self.node_index = index
        self.node_norm_responses = []

    def __str__(self):
        return str(self.node_index)

    def __repr__(self):
        return self.node_index

    @property
    def shape(self):
        return np.array(self.node_norm_responses).shape


class ImpulseModel():
    def __init__(self, N=100, dt=0.1, show_progress=True):
        self.N = N
        self.dt = dt
        self.impulse_responses = []

        if show_progress:
            self.rangefunc = trange
        else:
            self.rangefunc = range

    def apply_impulse_lti(self, adjmat, magnitude=1.0,
                          N=None, dt=None, stabilize=False):
        if N is None:
            N = self.N
        if dt is None:
            dt = self.dt

        A = adjmat

        if stabilize:
            A = self._stabilize_matrix(A, m=1.0)

        numchans, _ = adjmat.shape

        impulse_responses = []

        # add input to each channel
        for j in range(numchans):
            # create input matrix
            B = np.zeros((numchans, 1))
            B[j] = magnitude

            # create state space model
            C = np.eye(numchans)
            D = np.zeros((numchans, 1))

            # define state-space model - discrete time system
            system = scipy.signal.StateSpace(A, B, C, D, dt=dt)

            # computes impulse response of the dtlti system
            x0 = 0
            time, y = system.impulse(x0=x0, t=None, n=N)

            # get the result in array form (NxC)
            y = y[0]

            # print(y.shape)
            # transpose to become CxN
            y = y.T

            # actually append the impulse responses
            impulse_responses.append(y)

        return np.array(impulse_responses)

    def run(self, adjmats, magnitude=1, stabilize=False):
        # get the shape of the data
        numwins, numchans, _ = adjmats.shape

        # initialize arrays to store norms of the impulse model
        # frag_inf_arr = np.zeros((numchans, numwins))
        # frag_1_arr = np.zeros((numchans, numwins))
        # frag_2_arr = np.zeros((numchans, numwins))

        # creaate a list of the impulse responses
        l2norm_responses = []

        self.all_impulse_responses = []

        # loop through all windows
        for i in self.rangefunc(numwins):
            adjmat = adjmats[i, ...].squeeze()

            if stabilize:
                adjmat = self._stabilize_matrix(adjmat, m=1.0)

                # if i == 0 :
                #     print("Stabilizing matrix! {} to eigenvalue {}".format(adjmat.shape, 1.0))

            # if i == 0:
            # print(adjmat.shape)

            # apply impulse to this lti model
            impulse_responses = self.apply_impulse_lti(
                adjmat, magnitude, self.N)

            # compute metrics on the impulse responses
            frag_inf, frag_1, frag_2 = self.compute_response_metrics(
                impulse_responses)

            # store all impulse responses per window
            self.all_impulse_responses.append(impulse_responses)

            # curr_impulse_responses = []
            # for j in range(len(frag_2)):
            #     # create impulse response object that is a tree
            #     impulse_response = ImpulseResponse(j)
            #     impulse_response.node_norm_responses = frag_2[j]
            #
            #     curr_impulse_responses.append(impulse_response)

            # frag_inf_arr[:, i] = np.array(frag_inf)
            # frag_1_arr[:, i] = np.array(frag_1)
            # frag_2_arr[:, i] = np.array(frag_2)

            l2norm_responses.append(frag_2)
            # print(np.array(frag_2).shape)

            # break
        return l2norm_responses

    def compute_response_metrics(self, impulse_responses):
        frag_inf = []
        frag_1 = []
        frag_2 = []

        for idx, y in enumerate(impulse_responses):
            # if idx == 0:
            #     print("Response metrics original shape: ", y.shape)

            # define different metrics
            frag_inf.append(np.linalg.norm(x=y, ord=np.inf, axis=1))
            frag_1.append(np.linalg.norm(x=y, ord=1, axis=1))
            frag_2.append(np.linalg.norm(x=y, ord=2, axis=1))

        return frag_inf, frag_1, frag_2

    def _stabilize_matrix(self, A, m):
        """
        function to stabilize the matrix, A based on its eigenvalues.

        Assumes discrete-time linear system.
        """

        # check if there are unstable eigenvalues for this matrix
        if np.sum(np.abs(np.linalg.eigvals(A)) > 1) > 0:
            # perform eigenvalue decomposition
            eigA, V = scipy.linalg.eig(A)

            # get magnitude of these eigenvalues
            abs_eigA = np.abs(eigA)

            # get the indcies of the unstable eigenvalues
            unstable_inds = np.where(abs_eigA > 1.)[0]

            # move all unstable evals to magnitude m
            for ind in unstable_inds:
                # extract the unstable eval and magnitude
                unstable_eig = eigA[ind]
                this_abs_eig = abs_eigA[ind]

                # compute scaling factor and rescale eval
                k = m / this_abs_eig
                eigA[ind] = k * unstable_eig

            # recompute A
            eigA_mat = np.diag(eigA)
            Aprime = np.dot(V, eigA_mat)
            A = scipy.linalg.lstsq(V.T, Aprime.T)[0].T

        return A
