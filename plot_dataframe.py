from lr_main import *

df=pd.read_csv(r'C:\Users\Tommy\AppData\Local\Programs\Python\Python37\github\ml-projects\lr\data\train.csv')

def plot_data(df):
    data_proc = data_processor(df, 'SalePrice')
    data_proc.process_df(special_columns=['LotFrontage','MasVnrArea','GarageYrBlt'])
    data_proc.view_dataframe(df.columns[1:10].tolist(), 6, 4, 'Images', 'tmp_doc.docx')

if __name__ == '__main__':
    plot_data(df)
    