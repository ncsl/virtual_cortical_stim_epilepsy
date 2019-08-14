from cortstim.base.utils.log_error import initialize_logger
from cortstim.base.config.masterconfig import Config
import cortstim.base.config.model_constants as constants


class BaseWindowModel(object):
    """
    Base class for a sliding window-based model of data.

    TODO:
        1. add future logs to temp dir and figure dir

    """

    def __init__(self, config=None):
        self.config = config or Config()
        # initializes the logger to output files to FOLDER_LOG
        self.logger = initialize_logger(
            self.__class__.__name__, self.config.out.FOLDER_LOGS)

    def __str__(self):
        return "Window Model"

    def __repr__(self):
        return "Window Model"


class PerturbationModel(BaseWindowModel):
    def __init__(self, radius, perturbtype):
        super(PerturbationModel, self).__init__()
        perturbtype = perturbtype.lower()
        if perturbtype not in [constants.COLUMN_PERTURBATION, constants.ROW_PERTURBATION]:
            raise ValueError("Perturbation type can only be {}, or {} for now. You "
                             "passed in {}.".format(constants.COLUMN_PERTURBATION,
                                                    constants.ROW_PERTURBATION,
                                                    perturbtype))
        self.radius = radius
        self.perturbtype = perturbtype

    @property
    def parameters(self):
        return (self.radius, self.perturbtype)

    def __str__(self):
        return "Perturbation Model {}".format(self.parameters)
