from cortstim.base.config.config import GenericConfig, FiguresConfig, \
    InputConfig, OutputConfig


class Config(object):
    def __init__(self,
                 raw_data_folder=None,
                 output_base=None,
                 separate_by_run=False):
        self.generic = GenericConfig()
        self.figures = FiguresConfig()

        self.input = InputConfig(raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)
