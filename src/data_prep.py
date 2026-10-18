import pandas as pd

##loading the data 
import pandas as pd

def load_raw_data(path1: str, path2: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_excel(path1)
    df2 = pd.read_excel(path2)
    return df1, df2







def clean_mostly_null_cols(df)-> pd.DataFrame:
    a=[]
    for i in df.columns:
        if df[df[i]==-99999].shape[0]>10000:
            a.append(i)
    return df.drop(a,axis=1)

def clean_mostly_null_rows(df)-> pd.DataFrame:
    for i in df.columns:
        df=df[df[i]!=-99999]
    return df

    


def find_merge_cols(df1,df2):
    merge_cols=[]
    for i in df1.columns.tolist():
        if i in df2.columns.tolist():
            merge_cols.append(i)
    return merge_cols
