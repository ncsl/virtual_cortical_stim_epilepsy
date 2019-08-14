import argparse
import sys

sys.path.append('../../../')
from cortstim.edp.loaders.dataset.clinical.excel_meta import ExcelReader

parser = argparse.ArgumentParser(prog='Preformatting clinical excel file.', description='')
parser.add_argument('rawexceldatafilepath', default="",
                    help='The filepath for the excel file to help merge and augment our metadata')
parser.add_argument('exceldatafilepath', default="",
                    help='The filepath for the excel file to help merge and augment our metadata')

if __name__ == '__main__':
    args = parser.parse_args()
    rawexcelfilepath = args.rawexceldatafilepath
    excelfilepath = args.exceldatafilepath

    clinreader = ExcelReader(rawexcelfilepath, expanding_semio=True)
    clinreader.write_to_excel(excelfilepath)