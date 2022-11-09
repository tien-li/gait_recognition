
import pandas as pd
def csv_to_xlsx_pd():
csv = pd.read_csv('/Users/ken/Desktop/專題/20200428/experiment_data0428/林天立.output/Normal_right.csv', encoding='utf-8')
csv.to_excel('/Users/ken/Desktop/專題/20200428/experiment_data0428/林天立.output/Normal_right.xlsx', sheet_name='Data')
if __name__ == '__main__':
csv_to_xlsx_pd()
