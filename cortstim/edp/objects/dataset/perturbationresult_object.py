# -*- coding: utf-8 -*-
import numpy as np

from cortstim.edp.objects.dataset.result_object import Result


class ImpulseResult(Result):
    def __init__(self, impulse_responses, metadata):
        super(ImpulseResult, self).__init__(impulse_responses, metadata)

        if impulse_responses.ndim != 3:
            raise AttributeError(
                "The passed in impulse model needs to have only 3 dimensions!")

        self.json_fields = [
            'onsetwin',
            'offsetwin',
            'resultfilename',
            'winsize',
            'stepsize',
            'radius',
            'perturbtype',
        ]

    def __str__(self):
        return "{} {} Impulse Response Model {}".format(self.patient_id,
                                                        self.dataset_id,
                                                        self.shape)

    @staticmethod
    def compute_fragilitymetric(minnormpertmat):
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
        minmax_fragilitymat = -1 * np.true_divide((minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
                                                  np.matlib.repmat(maxacrosstime - minacrosstime, N, 1))
        return minmax_fragilitymat

    def likelihood_spread(self, impulse_responses):
        """
        Function that takes in a list of impulse responses from a certain node, and rank orders them based on their norm
        responses.

        :param impulse_responses:
        :return:
        """
        # sort from greatest to least
        sorted_inds = np.argsort(impulse_responses, axis=-1, kind='mergesort')

        # apply min-max normalization along the column

        # loop through each node's response to the impulse
        for i, response in enumerate(impulse_responses):
            print(i, response.shape)

    def compute_selfnorm(self):
        """
        Function to compute the norm of the node's own response to its impulse response. This
        is essentially a measure of how unstable a node is to it's own auto response.

        :return:
        """
        pass

