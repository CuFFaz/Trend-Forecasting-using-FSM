import time
import pandas as pd
import argparse
from model.model import TrendFSMModel

class ModelMain:
    def __init__(self, keyword) -> None:
        df = pd.read_csv('Data/newscycle_daily.csv', parse_dates=['date'])
        self.df = df[df['keyword'] == keyword]
        self.fsm = TrendFSMModel(keyword=keyword, df_slopes=df)

    def TrendsSoFar(self):
        return self.fsm.process_data_TD(df=self.df)
        
    # def CurrentState(self):
        # print("Current State Forecasted:")
    #     self.TrendsSoFar()
        # return self.fsm.state

    def NextState(self):
        print("Next States Forecasted:")
        self.TrendsSoFar()
        return self.fsm.state
      
    def TimeToPeak(self): #WIP
        return

    def TimeToDie(self): #WIP
        return

    @property
    def is_Dead(self):
        return
        
if __name__ == "__main__":
    t1 = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, help='input keyword which is to be predicted')
    opt = parser.parse_args()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)
    app = ModelMain(keyword=opt.keyword)

    print(app.NextState())
