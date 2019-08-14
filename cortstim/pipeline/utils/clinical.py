import sys

sys.path.append('../../')

from cortstim.edp.loaders.dataset.clinical.excel_meta import ExcelReader


def load_clinical_df(excelfilepath):
    excelreader = ExcelReader(filepath=excelfilepath)
    excelreader.read_formatted_df(excelfilepath)
    clindf = excelreader.ieegdf.clindf
    return clindf


