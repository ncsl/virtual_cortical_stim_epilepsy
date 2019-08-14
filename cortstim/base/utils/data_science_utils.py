import collections

import numba
import numpy as np
import numpy.matlib
from sklearn.preprocessing import MinMaxScaler


@numba.jit
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def get_elbows(d, n=3, threshold=False):
    """
    Given a decreasingly sorted vector, return the given number of elbows.

    Reference:
    ##   Zhu, Mu and Ghodsi, Ali (2006), "Automatic dimensionality selection from
    ##   the scree plot via the use of profile likelihood", Computational
    ##   Statistics & Data Analysis

    :param d:
    :param n:
    :param threshold: (float)
    :return: q: (np.ndarray) a vector of length n.
    """

    def gauss_logL(xbar, V, n, sigma, mu):
        """Equation 5.57: gaussian likelihood"""
        return (-(n + 1) * np.log(sigma)
                - 0.5 * n * ((xbar - mu) ** 2 + V) / sigma ** 2)

    # check if d is sorted
    if not is_sorted(d):
        raise RuntimeError("Passed in vector should be sorted in decreasing order!")

    # apply threshold or not
    if threshold:
        d = d > threshold

    p = len(d)
    if p == 0:
        raise RuntimeError(f"d should have elements larger then the threshold {threshold}.")

    # run log-likelihood algorithm
    loghood = []
    for i, q in enumerate(d):
        # compute avg
        mu = np.mean(d[:i])
        mu2 = np.mean(d[i:])
        sigma2 = np.var(d[:i])

        n = len(d[:i])
        m = len(d[i:])

        loghood.append(gauss_logL(d[:i], 0, n, sigma2, mu) +
                       gauss_logL(d[i:], 0, m, sigma2, mu2))

    # get the maximimum occurrence of log-likelihood
    maxind = np.where(loghood == np.max(loghood))[0]

    if n > 1 and q < p:
        return
    else:
        return q


def cutoff_youdens(fpr, tpr, thresholds):
    """
    Returns the index along fpr/tpr at which youden stat is maximized.

    :param fpr:
    :param tpr:
    :param thresholds:
    :return:
    """
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def apply_thresholding(mat, threshold):
    mat[mat < threshold] = 0.
    return mat


def compute_ez_likelihood(cezmat, oezmat):
    meancez = np.nanmean(cezmat, axis=1, keepdims=True)
    meanoez = np.nanmean(oezmat, axis=1, keepdims=True)

    # preprocess = StandardScaler(with_mean=False)
    preprocess = MinMaxScaler()
    preprocess.fit(np.concatenate((meancez, meanoez)))
    meancez = preprocess.transform(meancez)
    meanoez = preprocess.transform(meanoez)

    ezratio = meancez / (meanoez + meancez)
    ezratio[np.isinf(ezratio)] = np.nan
    ezratio[np.isnan(ezratio)] = np.nanmax(ezratio)
    return np.squeeze(ezratio)


def compute_fragilityratio(cezmat, oezmat):
    # print(cezmat.shape, oezmat.shape)
    meancez = np.nanmean(cezmat, axis=1, keepdims=True)
    meanoez = np.nanmean(oezmat, axis=1, keepdims=True)
    # self.cezmat += 1e-4
    # self.oezmat += 1e-4
    # meancez = scipy.stats.mstats.gmean(self.cezmat, axis=1)[:,np.newaxis]
    # meanoez = scipy.stats.mstats.gmean(self.oezmat, axis=1)[:,np.newaxis]

    preprocess = MinMaxScaler()
    preprocess.fit(np.concatenate((meancez, meanoez)))
    meancez = preprocess.transform(meancez)
    meanoez = preprocess.transform(meanoez)

    fratio = meancez / (meanoez + meancez)
    fratio[np.isinf(fratio)] = np.nan
    fratio[np.isnan(fratio)] = np.nanmax(fratio)
    return np.squeeze(fratio)


def compute_fragilitymetric(minnormpertmat):
    # get dimensions of the pert matrix
    N, T = minnormpertmat.shape
    # assert N < T
    fragilitymat = np.zeros((N, T))
    for icol in range(T):
        fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) / \
                                np.max(minnormpertmat[:, icol])
    return fragilitymat


def compute_fragilitymetric_inv(minnormpertmat):
    # get dimensions of the pert matrix
    N, T = minnormpertmat.shape
    minnormpertmat = 1 - minnormpertmat
    # assert N < T
    fragilitymat = np.zeros((N, T))
    for icol in range(T):
        fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) / \
                                np.max(minnormpertmat[:, icol])
    return fragilitymat


def compute_minmaxfragilitymetric(minnormpertmat):
    # get dimensions of the pert matrix
    N, T = minnormpertmat.shape
    # assert N < T
    minmax_fragilitymat = np.zeros((N, T))

    # get the min/max for each column in matrix
    minacrosstime = np.min(minnormpertmat, axis=0)
    maxacrosstime = np.max(minnormpertmat, axis=0)

    # normalized data with minmax scaling
    minmax_fragilitymat = -1 * np.true_divide((minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
                                              np.matlib.repmat(maxacrosstime - minacrosstime, N, 1))
    return minmax_fragilitymat


def compute_znormalized_fragilitymetric(minnormpertmat):
    # get dimensions of the pert matrix
    N, T = minnormpertmat.shape

    # get mean, std
    avg_contacts = np.mean(minnormpertmat, keepdims=True, axis=0)
    std_contacts = np.std(minnormpertmat, keepdims=True, axis=0)

    # normalized data with minmax scaling
    return (minnormpertmat - avg_contacts) / std_contacts


def split_inds_outcome(patlist, mastersheet):
    succ_inds = []
    fail_inds = []

    for i, pat in enumerate(patlist):
        outcome = mastersheet.get_patient_outcome(pat)
        if outcome == 's':
            succ_inds.append(i)
        elif outcome == 'f':
            fail_inds.append(i)

    return succ_inds, fail_inds


def split_inds_engel(patlist, mastersheet):
    engel_inds = collections.defaultdict(list)
    for i, pat in enumerate(patlist):
        engel_score = mastersheet.get_patient_engelscore(pat)
        engel_inds[engel_score].append(i)
    return engel_inds

def split_inds_ilae(patlist, mastersheet):
    engel_inds = collections.defaultdict(list)
    for i, pat in enumerate(patlist):
        engel_score = mastersheet.get_patient_ilaescore(pat)
        engel_inds[engel_score].append(i)
    return engel_inds


def split_inds_modality(patlist, mastersheet):
    modality_inds = collections.defaultdict(list)
    for i, pat in enumerate(patlist):
        modality = mastersheet.get_patient_modality(pat)
        modality_inds[modality].append(i)
    return modality_inds


def get_numerical_outcome(patlist, mastersheet):
    ytrue = []
    for i, pat in enumerate(patlist):
        outcome = mastersheet.get_patient_outcome(pat)
        if outcome == 's':
            ytrue.append(1)
        elif outcome == 'f':
            ytrue.append(0)
    return ytrue


def split_inds_clindiff(patlist, mastersheet):
    clindiff_inds = collections.defaultdict(list)
    for i, pat in enumerate(patlist):
        clindiff = mastersheet.get_patient_clinicaldiff(pat)
        clindiff_inds[clindiff].append(i)
    return clindiff_inds
