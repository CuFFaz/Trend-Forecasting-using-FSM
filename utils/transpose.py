import pandas as pd

class TransformDF:

    def __init__(self, df, type, kwdf) -> None:
        self.df = df
        self.type = type
        self.kwdf = kwdf

    def transform(self):

        mt_df = pd.DataFrame(columns=['date', 'type', 'category',  'keyword', 'value'])
        class_dict = dict(zip(self.kwdf.keyword, self.kwdf.categories))
        numb = 1
        df = self.df.iloc[5:].reset_index(drop=True)
        for i in range(len(df)):
            for j in range(len(df.loc[i])):
                try:
                    ls = []
                    kw = df.iloc[i].index[j+1]
                    ls.append(df.iloc[i][0])
                    ls.append(type)
                    ls.append(class_dict.get(kw))        
                    ls.append(kw)
                    ls.append(df.iloc[i][j+1])
                    mt_df.loc[len(mt_df.index)] = ls
                    print(numb)
                    numb += 1
                except IndexError:
                    pass
        sheet_name = 'Data/newscycle_'+ type +'.csv'
        mt_df.to_csv(sheet_name, index=False)

