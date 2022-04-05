import pytz
import pandas as pd
from pytrends.request import TrendReq
import calendar
import time
import datetime
import numpy as np
import warnings

from utils.transpose import TransformDF

warnings.filterwarnings("ignore")

class Trends:

    def __init__(self, kw_list, key_ref):
        self.UTC_tmz = pytz.utc
        self.IST_tmz = pytz.timezone('Asia/Kolkata')
        self.key_ref = "google news"
        self.year = 2021

        self.kw_list = kw_list
        self.key_ref = key_ref
        self.COUNTRY = "IN"
        self.CATEGORY = 0
        self.SEARCH_TYPE = ''

    def get_yearly_timescale(self):
        date_intervals_ls = []

        for mon in range(1,2):
            _, num_days = calendar.monthrange(self.year, mon)
            date_intervals_ls.append(datetime.date(self.year, mon, 1).strftime("%Y-%m-%d")) #First day of the month
            date_intervals_ls.append(datetime.date(self.year, mon, num_days).strftime("%Y-%m-%d")) #Last day of the month
        
        return date_intervals_ls

    def get_normalised(self, df_kwset_interval):
        dates_col = df_kwset_interval['date']
        #Replace zeroes in key_ref by 1 for Normalisation
        df_kwset_interval[self.key_ref] = df_kwset_interval[key_ref].replace(0, 1)
        #Normalise with respect to key_ref
        df_kwset_interval.drop('date',axis=1, inplace=True)
        for col in df_kwset_interval:
            df_kwset_interval[col] = df_kwset_interval[col]/df_kwset_interval[key_ref]
        df_kwset_interval.reset_index()
        #Drop key_ref column
        df_kwset_interval.drop(self.key_ref, axis=1, inplace=True)

        return df_kwset_interval, dates_col

    def get_trends(self, i, kw_list, date_intervals_ls):

        pytrends = TrendReq(hl='en-US', tz=330)

        #Loop through the keywords in SET OF 4 
        kwset_df_ls = [] #List to concat results of 4 keywords
        for cal in range(0, len(date_intervals_ls),2): #loop through months
            try:
                date_instance = date_intervals_ls[cal]+" "+date_intervals_ls[cal+1]
                l = kw_list[i:(i+4)]
                l.append(self.key_ref) #Add up key_ref to set of kws for relative scores wrt to key_red
                print("Busy Requesting data.....")
                pytrends.build_payload(l,
                                timeframe = date_instance, 
                                geo=self.COUNTRY, 
                                cat=self.CATEGORY,
                                gprop=self.SEARCH_TYPE)
                df_kwset = pytrends.interest_over_time()
                df_kwset.reset_index(inplace=True)
                df_kwset.drop('isPartial', axis = 1, inplace=True)

                kwset_df_ls.append(df_kwset)

            except IndexError:
                pass

        df_kwset_interval = pd.concat(kwset_df_ls) #Form up resultant data frame of 4 Keywords
        raw_name = "raw_dataset"+str("_")+str((i+4)//4)+".csv" #Serial naming
        df_kwset_interval.to_csv(raw_name, index = False) #Get raw csv
        return df_kwset_interval

    def get_concatenated_from_list(self, df_concat_ls):
        maj_dataframe = pd.concat(df_concat_ls, axis=1) #Form up resultant dataframe of all Keywords
        return maj_dataframe

    def get_concatenated_dates(self, maj_dataframe, dates_col):
        maj_dataframe = pd.concat((dates_col, maj_dataframe), axis=1) #Join the dates column
        # maj_dataframe.to_csv('concatenated_maj_df.csv', index=False) #Get resultant dataframe in csv
        return maj_dataframe

    #Time conversion from utc time to local time if required
    # def datetime_UTC_to_IST(self):
    #     df['date'] = pd.to_datetime(self.dataframe['date'])
    #     df['date'].dt.tz_localize(self.UTC_tmz).dt.tz_convert(self.IST_tmz).dt.tz_localize(None)

    def get_minimized(self, maj_dataframe, dates_col):
        maj_dataframe.drop('date',axis=1, inplace=True)
        for s_col in maj_dataframe:
            dtfm = np.array(maj_dataframe[s_col].values.tolist()) #Line it up in an array
            maj_dataframe[s_col] = np.where(dtfm < 0.1, 0.1, dtfm).tolist()

        fin_dataframe = pd.concat((dates_col,maj_dataframe), axis=1) #Array to dataframe
        return fin_dataframe

    def get_aggregate(self, fin_dataframe):
        return fin_dataframe.set_index('date').rolling(window=5).mean().to_csv('Data/trends_data.csv') #Get resultant dataframe with log scale and
                                                                             #moving average in csv

    def process_data(self):

        df_concat_ls = []
        i = 0
        date_intervals_ls = self.get_yearly_timescale()
        while i < len(kw_list):
            df_kwset_interval = self.get_trends(i, self.kw_list, date_intervals_ls)
            df_kwset_interval_norm, dates_col = self.get_normalised(df_kwset_interval)
            df_concat_ls.append(df_kwset_interval_norm)
            #Increment to index the next 4 keywords
            i += 4
            print("Sleeping...")
            time.sleep(5) #sleep to avoid timeouts
            print('Awake!')
            j = str((i+4)//4)
            print("Keyword set No.{0} is running".format(j))

        maj_dataframe = self.get_concatenated_from_list(df_concat_ls)
        maj_dataframe = self.get_concatenated_dates(maj_dataframe, dates_col)    
        fin_dataframe = self.get_minimized(maj_dataframe, dates_col)
        fin_dataframe = self.get_aggregate(fin_dataframe)

        return fin_dataframe

if __name__ == "__main__":

    #Read csv where in all our Keywords are mentioned
    df_kw = pd.read_csv('keywords.csv')
    data_duration_type = 'daily'
    kw_list = df_kw['keywords'].to_list()
    key_ref = "google news"
    #Get Keyword specific Data from Google Trends 
    a = Trends(kw_list=kw_list, key_ref=key_ref)
    final_df = a.process_data()

    #Taking Transpose of the whole dataframe to a better suitable structure to feed to FSM Model
    a = TransformDF(df=final_df, type=data_duration_type, kwdf=df_kw)
    a.transform()