from codecs import ignore_errors
from datetime import datetime
from enum import Enum
from transitions import Machine
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")

class TrendState(Enum):
    STEADY_GROWTH = 'steady_growth'
    INITIAL_INTEREST = 'initial_interest'
    ACCELERATED_INTEREST = 'accelerated_interest'
    SPIKE = 'spike'
    PEAK = 'peak'
    VALLEY = 'valley'
    PLATEAU = 'plateau'
    STEADY_DECLINE = 'steady_decline'
    INITIAL_DECLINE = 'initial_decline'
    GRADUAL_DEATH = 'gradual_death'
    CRASH = 'crash'

class TimeFrame(Enum):
    HOUR = 'hourly'
    DAY = 'daily'
    WEEK = 'weekly'

class ModelParams(object):
    df = pd.read_csv('Data/Train/Slopes.csv')
    ASC_10_QUANTILE = df['asc_slope'].quantile(0.1)
    ASC_25_QUANTILE = df['asc_slope'].quantile(0.25)
    ASC_50_QUANTILE = df['asc_slope'].quantile(0.5)
    ASC_75_QUANTILE = df['asc_slope'].quantile(0.75)
    DSC_25_QUANTILE = df['dsc_slope'].quantile(0.25)
    DSC_50_QUANTILE = df['dsc_slope'].quantile(0.5)
    DSC_75_QUANTILE = df['dsc_slope'].quantile(0.75)
    DSC_90_QUANTILE = df['dsc_slope'].quantile(0.1)


    # Lower bound included anFSM Predictor for d upper bound not included
    SLOPE_ASCEND_1 = {"lower": 0.0, "upper": ASC_25_QUANTILE}
    SLOPE_ASCEND_2 = {"lower": ASC_25_QUANTILE, "upper": ASC_50_QUANTILE}
    SLOPE_ASCEND_3 = {"lower": ASC_50_QUANTILE, "upper": ASC_75_QUANTILE}
    SLOPE_SPIKE = {"lower": ASC_75_QUANTILE, "upper": 100}

    SLOPE_DESCEND_3 = {"lower": DSC_75_QUANTILE, "upper": DSC_90_QUANTILE}
    SLOPE_DESCEND_2 = {"lower": DSC_50_QUANTILE, "upper": DSC_75_QUANTILE}
    SLOPE_DESCEND_1 = {"lower": DSC_25_QUANTILE, "upper": DSC_50_QUANTILE}
    SLOPE_CRASH = {"lower": -100, "upper": DSC_25_QUANTILE}

    df = pd.read_csv('Data/Train/TrendDurations.csv')
    ASC_TREND_DUR_MEAN = df['asc_dur'].mean()
    DSC_TREND_DUR_MEAN = df['dsc_dur'].mean()
    DSC_TREND_DUR_75_QUANTILE = df['dsc_dur'].quantile(0.75)

TREND_THRESHOLD = 0.2 # Percent of maximum value considered as threshold

# ascend_1 < ascend_2 < ascend_3 < ascend_spiked
# abs_value of: descend_crashed > descend_1 > descend_2 > descend_3
TRANSITIONS = [
                {'trigger':'reset_trend', 'source':'*', 'dest':'valley'},
                # {'trigger':'ascend_1', 'source':'steady_growth', 'dest':'steady_growth'},
                # {'trigger':'ascend_1', 'source':'accelerated_interest', 'dest':'accelerated_interest'},
                # {'trigger':'ascend_2', 'source':'accelerated_interest', 'dest':'accelerated_interest'},
                # {'trigger':'ascend_1', 'source':'spike', 'dest':'spike'},
                # {'trigger':'ascend_2', 'source':'spike', 'dest':'spike'},
                # {'trigger':'ascend_3', 'source':'spike', 'dest':'spike'},
                {'trigger':'ascend_1', 'source':'*', 'dest':'initial_interest'},
                {'trigger':'ascend_2', 'source':'*', 'dest':'steady_growth'},
                {'trigger':'ascend_3', 'source':'*', 'dest':'accelerated_interest'},
                {'trigger':'ascend_spiked', 'source':'*', 'dest':'spike'},
                
                {'trigger':'peaked', 'source':'*', 'dest':'peak'},

                {'trigger':'descend_1', 'source':'steady_decline', 'dest':'steady_decline'},
                {'trigger':'descend_1', 'source':'gradual_death', 'dest':'gradual_death'},
                {'trigger':'descend_2', 'source':'gradual_death', 'dest':'gradual_death'},
                {'trigger':'descend_1', 'source':'crash', 'dest':'crash'},
                {'trigger':'descend_2', 'source':'crash', 'dest':'crash'},
                {'trigger':'descend_3', 'source':'crash', 'dest':'crash'},
                {'trigger':'descend_1', 'source':'*', 'dest':'initial_decline'},
                {'trigger':'descend_2', 'source':'*', 'dest':'steady_decline'},
                {'trigger':'descend_3', 'source':'*', 'dest':'gradual_death'},
                {'trigger':'descend_crashed', 'source':'*', 'dest':'crash'},

                {'trigger':'descend_valleyed', 'source':'*', 'dest':'plateau'},
                {'trigger':'descend_1', 'source':'plateau', 'dest':'plateau'},
                {'trigger':'descend_2', 'source':'plateau', 'dest':'plateau'},
                {'trigger':'descend_3', 'source':'plateau', 'dest':'plateau'},
                {'trigger':'descend_crashed', 'source':'plateau', 'dest':'plateau'},
            ]


class TrendFSMModel(object):

    states = [state.value for state in TrendState]

    def __init__(self, keyword: str, df_slopes: pd.DataFrame, timeframe=TimeFrame.DAY):
        
        # Keyword which is being modelled
        self.keyword = keyword
        self.timeframe = timeframe

        # 
        # '; track of data points so far in pandas dataframe
        self.slopes_df = df_slopes
        self.peak_trend_value = 0.0
        self.trend_min_value = 100
        self.peak_count = 0
        self.asc_dur = 0
        self.dsc_dur = 0

        # Initialize the state machine
        self.machine = Machine(model=self, states=TrendFSMModel.states, initial=TrendState.VALLEY.value, transitions=TRANSITIONS)
        self.pre_peak_states = [TrendState.VALLEY.value, 
                                TrendState.STEADY_GROWTH.value, 
                                TrendState.INITIAL_INTEREST.value, 
                                TrendState.ACCELERATED_INTEREST.value,
                                TrendState.SPIKE.value,]

    @property
    def time_elapsed(self):
        return self.slopes_df.shape[0]

    @property
    def past_peak(self):
        if self.state in self.pre_peak_states:
            return False
        else:
            return True

    # Returns latest state
    def process_current_point(self, ascending_slope: float=None, descending_slope: float=None, trend_value: float=0):
        if 0 < self.peak_trend_value < trend_value:
            self.peak_count += 1

        if not self.past_peak and descending_slope is not None:
            self.peaked()
            self.peak_trend_value = trend_value

        if trend_value < self.trend_min_value:
            self.trend_min_value = trend_value

        if ascending_slope:
            if ModelParams.SLOPE_ASCEND_1['lower'] <= ascending_slope < ModelParams.SLOPE_ASCEND_1['upper']:
                self.ascend_1()
            if ModelParams.SLOPE_ASCEND_2['lower'] <= ascending_slope < ModelParams.SLOPE_ASCEND_2['upper']:
                self.ascend_2()
            if ModelParams.SLOPE_ASCEND_3['lower'] <= ascending_slope < ModelParams.SLOPE_ASCEND_3['upper']:
                self.ascend_3()
            if ModelParams.SLOPE_SPIKE['lower'] <= ascending_slope <= ModelParams.SLOPE_SPIKE['upper']:
                self.ascend_spiked()
        
        if descending_slope:
            if ModelParams.SLOPE_DESCEND_1['lower'] <= descending_slope < ModelParams.SLOPE_DESCEND_1['upper']:
                self.descend_1()
            if ModelParams.SLOPE_DESCEND_2['lower'] <= descending_slope < ModelParams.SLOPE_DESCEND_2['upper']:
                self.descend_2()
            if ModelParams.SLOPE_DESCEND_3['lower'] <= descending_slope < ModelParams.SLOPE_DESCEND_3['upper']:
                self.descend_3()
            if ModelParams.SLOPE_CRASH['lower'] <= descending_slope <= ModelParams.SLOPE_CRASH['upper']:
                self.descend_crashed()
            
        return self.state

    def process_data_TD(self, df: pd.DataFrame):
        if df.empty:
            return 'No Data'

        max_val = df['value'].max()
        df = df[df['value'] > max_val * TREND_THRESHOLD]
        if df.empty:
            return 'No significant trend found'
        self.slopes_df = df

        # Reset Trend before processing
        self.reset_trend()

        max_index = df['value'].idxmax()

        df_ascending = df[df['date'] <=  df['date'].loc[max_index]]
        df_ascending['series_count'] = np.arange(df_ascending.shape[0])
        df_ascending['ascending_slope'] = df_ascending.apply(lambda row: 
                                            (row['value'] - df_ascending['value'].iloc[0]) / row['series_count'] if row['series_count'] > 0 else 0, axis=1)

        df_ascending['trend_class'] = ''
        for index, row in df_ascending.iterrows():
            if row['series_count'] == 0:
                self.ascend_1()
            df_ascending.at[index, 'trend_class'] = self.process_current_point(ascending_slope=row['ascending_slope'])
        
        self.asc_dur = df_ascending.shape[0]
        # print(df_ascending)

        df_descending = df[df['date'] >=  df['date'].loc[max_index]]
        df_descending['series_count'] = np.arange(df_descending.shape[0])
        df_descending['descending_slope'] = df_descending.apply(lambda row: 
                                            (row['value'] - df_descending['value'].iloc[0]) / row['series_count'] if row['series_count'] > 0 else 0, axis=1)

        df_descending['trend_class'] = ''
        if df_descending.shape[0] > 1:
            for index, row in df_descending.iterrows():
                df_descending.at[index, 'trend_class'] = self.process_current_point(descending_slope=row['descending_slope'], trend_value=row['value'])
                if row['series_count'] > ModelParams.DSC_TREND_DUR_75_QUANTILE:
                    self.descend_valleyed()
        
        self.dsc_dur = df_descending.shape[0]
        # print(df_descending)
        
        df_ascending = df_ascending.rename({'ascending_slope': 'slope'}, axis=1)
        df_descending = df_descending.rename({'descending_slope': 'slope'}, axis=1)
        df = df_ascending.append(df_descending, ignore_index=True)
        print(df.drop('date', 1))
        self.plot_trend_slopes(df=df)
        return

    def plot_trend_slopes(self, df: pd.DataFrame):
        fig1, (ax1, ax2) = plt.subplots(2)
        fig1.suptitle('Trend Curve')
        ax1.plot(df['date'], df['value'])
        ax2.plot(df['date'], df['slope'])
        plt.show()
        return

    def GetDataGoogleTrend(self) -> pd.DataFrame:
        pass

# WIP
class Prediction(TrendFSMModel):

    def __init__(self, keyword: str, df_slopes: pd.DataFrame, timeframe=TimeFrame.DAY):
        super().__init__(keyword, df_slopes, timeframe)

    def time_to_peak_or_die(self, state, slope):

        ascend = [TrendState.STEADY_GROWTH.value, TrendState.INITIAL_INTEREST.value, TrendState.ACCELERATED_INTEREST.value, TrendState.SPIKE.value]
        descend = [TrendState.STEADY_DECLINE.value, TrendState.INITIAL_DECLINE.value, TrendState.GRADUAL_DEATH.value, TrendState.CRASH.value]

        x = slope['datetime']
        y = slope['slope_value']
        degree=3 # Degree of polynomial, assuming 3 as in test it gave less deviations

        if state in ascend or state in descend:
            # For ascend, we are assuming latest point is pointing peak, we can assume that while approaching peak, slope is decreasing.
            # As in Descend, slope will be decreasing again 
            # Therefore, we can fit polynomial function

            # Getting coeffiecients assuming it as univariate function depending on slope_value only
            coefs = np.polyfit(x, y, degree)

            # For time_to_peak or die, we have to find point at which slope is near to zero
            roots = (np.poly1d(coefs) - 0).roots
            slope_zeroing_time = [root for root in roots if np.isreal(root) ][0].real
            return slope_zeroing_time

    def Max_Trend_Strength(self):
        pass

if __name__ == "__main__":
    t1 = time.perf_counter()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)

    # app = TrendFSMModel(keyword='Omicron', df_slopes=pd.DataFrame())
    # df = pd.read_csv('Data/newscycle_daily.csv', parse_dates=['date'])
    # df = df[df['keyword'] == 'Omicron']
    # #for i in range(df.shape[0]):
    #     #app.process_data_TD(df=df[:i+1])
    # app.process_data_TD(df=df)#[df['date'] < datetime(2021, 12, 19)])
    t2 = time.perf_counter()