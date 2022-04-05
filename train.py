import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from itertools import zip_longest

warnings.filterwarnings("ignore")

class TrainingData(object):
    def __init__(self, filename) -> None:
        self.df = pd.read_csv(filename, parse_dates=['date'])
        self.threshold_pct = 0.2
        self.train_data_folder = 'Data/Train/'
        super().__init__()

    def GetLocalMaxima(self, df):

        # Find local peaks
        df['max_val'] = df.value[(df.value.shift(10) < df.value) & 
                                (df.value.shift(-10) < df.value) & 
                                (df.value.shift(-2) < df.value) & 
                                (df.value.shift(2) < df.value)]
        threshold = df['value'].max() * 0.2
        df['max_val'] = df.value[(df.max_val > threshold)]
        df = df[df['value'] > threshold]
        # Plot results
        plt.scatter(df.index, df['max_val'], c='g')
        df.value.plot()
        plt.show()

    def KeyWordSlope(self, keyword):
        df = self.df[self.df['keyword'] == keyword]
        threshold = df['value'].max() * self.threshold_pct
        df = df[df['value'] > threshold]

        max_index = df['value'].idxmax()
        df_ascending = df[df['date'] <=  df['date'].loc[max_index]]
        df_ascending['series_count'] = np.arange(df_ascending.shape[0])
        df_ascending['ascending_slope'] = df_ascending.apply(lambda row: 
                                            (row['value'] - df_ascending['value'].iloc[0]) / row['series_count'] if row['series_count'] > 0 else 0, axis=1)

        df_ascending = df_ascending[df_ascending['ascending_slope'] > 0]

        df_descending = df[df['date'] >=  df['date'].loc[max_index]]
        df_descending['series_count'] = np.arange(df_descending.shape[0])
        df_descending['descending_slope'] = df_descending.apply(lambda row: 
                                            (row['value'] - df_descending['value'].iloc[0]) / row['series_count'] if row['series_count'] > 0 else 0, axis=1)
        df_descending = df_descending[df_descending['descending_slope'] < 0]

        if df_ascending['ascending_slope'].max() < 1:
            asc_slopes = []
        else:
            asc_slopes = df_ascending['ascending_slope'].tolist()
        if df_descending['descending_slope'].min() > -1:
            dsc_slopes = []
        else:
            dsc_slopes = df_descending['descending_slope'].tolist()
        
        return asc_slopes, dsc_slopes, df_ascending, df_descending

    def AllSlopesData(self):
        asc_slopes = []
        dsc_slopes = []

        for keyword in self.df['keyword'].unique():
            print(f'Estimating slopes for {keyword}')
            asc_slope, dsc_slope, df_asc, df_dsc = self.KeyWordSlope(keyword=keyword)
            asc_slopes = asc_slopes + asc_slope
            dsc_slopes = dsc_slopes + dsc_slope
            df = pd.DataFrame(columns=['asc_slope', 'dsc_slope'], data=zip_longest(asc_slopes, dsc_slopes, fillvalue=None))
        return df

    def TrendDuration(self):
        asc_dur = []
        dsc_dur = []

        for keyword in self.df['keyword'].unique():
            print(f'Estimating duration for {keyword}')

            asc_slope, dsc_slope, df_asc, df_dsc = self.KeyWordSlope(keyword=keyword)
            asc_dur = asc_dur + [df_asc.shape[0]]
            dsc_dur = dsc_dur + [df_dsc.shape[0]]
            df = pd.DataFrame(columns=['asc_dur', 'dsc_dur'], data=zip_longest(asc_dur, dsc_dur, fillvalue=None))
        return df

    def Train(self):
        df = self.AllSlopesData()
        df.to_csv(f'{self.train_data_folder}Slopes.csv')
    
        df = self.TrendDuration()
        df.to_csv(f'{self.train_data_folder}TrendDurations.csv')
        return
import os
if __name__ == "__main__":
    t1 = time.perf_counter()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)

    app = TrainingData(filename='Data/newscycle_daily.csv')
    app.Train()


    t2 = time.perf_counter()