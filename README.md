## Trend Forecasting using FSM

### Overview
Predicting trend patterns based on <a href='https://en.wikipedia.org/wiki/Finite-state_machine'>Finite State machine</a> via state-slope estimation. Trend patterns when considered as a state machine, shows the possibility of an arbitrary point on the trendline to possess one of the multiple states(based on its state transition table)(predefined in the script) through its slope and direction. 
For a brief understanding, for a certain trend 3 states can be considered - ascending, peak or descending state. Based on initial input i.e slope of a line, we map it to a particular state and later transitions between these states are estimated based on the training data of different variants of <a href='https://www.ig.com/en/trading-strategies/10-chart-patterns-every-trader-needs-to-know-190514'>trend patterns</a>. For such a data dependent model, we use data from <a href='https://trends.google.com/trends'>Google Trends</a> for past 365 days for 61 keywords. 

Vision to the Project: Extrapolate trends of unusual keywords whose polularity on Google Search just started amplifying swiftly. Example- Covid, Omicron, Someone slapped someone at Oscars, etc

Primarily this project has 3 parts:
1. Data Extraction and Preprocessing
2. Prepare Training Data
3. Fit Model

### Google Search based Data
<a href='https://trends.google.com/trends'>Google Trends</a> provides Periodic Trends for keywords in the form of normalised values which are structured in a beautiful manner. Trends of multiple keywords can be compared and the relative trend data is obtained. These values are respectively scaled upto the one which has highest search volumes among a set of upto 5 keywords. Now, GT doesn't let us compare for more than a set of 5 keywords at once, hence we normalise it ourselves via mathematical means. Thanks to <a href='https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa#:~:text=Currently%2C%20the%20public%2Dfacing%20Google,of%20all%20the%20major%20candidates.'>this</a>.
Every bit of the data was acquired from GT via <a href='https://pypi.org/project/pytrends/'>PyTrends</a>, an unconventional api for GT.

### Data Preprocessing
Data is gathered by scraping from pytrends and later pre-processed in trends.py and transform.py as follows: 
- Normalisation
- Concatenation 
- Minimization
- Moving averages (Moving Averages to smooth up the trend slopes and directions)
- Transform (Take Transpose of every individual datapoint and make it more model specific)

### Local Maxima, Slopes and Trend Durations
Find the local peaks such that we obtain only the trending pattern, from the trend's initial ascend till its death which is proportional to a threshold which is peak specific.
Slopes are calculated for these trending patterns and respective durations for slope ascend and descend are computed as well in train.py

### FSM Model
Implementation of FSM is done in model.py where-in States mentioned below are predefined for variants of trend patterns according to the slopes.

| Trend States    |
|----------------------|
| STEADY_GROWTH        |
| INITIAL_INTEREST     |
| ACCELERATED_INTEREST |
| SPIKE                |
| PEAK                 |
| VALLEY               |
| PLATEAU              |
| STEADY_DECLINE       |
| INITIAL_DECLINE      |
| GRADUAL_DEATH        |
| CRASH                |

Input to the Model: A single keyword.
Output of Model: Outputs can be channeled between TrendsSoFar, CurrentState, NextState in main.py
For a single input keyword,
- TrendsSoFar - Provides clipped off trend patterns obtained from training data for a specific keyword.
- CurrentState - Names of Assigned states for previous and present datapoints for a specific keyword. 
- NextState - Names of Future states predicted for a specific keyword.
- TimeTOPeak(WIP) - Estimated Time (in days) to reach the peak of the trend.
- TimeToDie(WIP) - Estimated Time (in days) for the trend to die from the peak to global minima. 

### To Run Locally...
1. Clone/Download the repo
```
$ git clone
```
2. Install dependencies for the project
```
$ pip3 install -r requirements.txt
```
3. Run trends.py for extracting daily data for past 365 days from google trends for keywords and their categories mentioned in 'keywords.csv' in root dir
```
$ python3 trends.py
```
4. Run train.py for computing slopes and trend durations which would be used by fsm model to define states.
```
$ python3 train.py
```
5. Run main.py with the keyword mentioned in the driver code whose trend is to be predicted.
```
$ python3 model.py
```

#### Tasks implemented in this project:
1. Scraping from PyTrends
2. Making multiple queries comparable in Google Trends
3. Data Preprocessing and Manipulating/Transforming Dataframes 
4. Slope Calculation
5. FSM Model implementation in python


#### References:

<a href='https://newslifespan.com/'>Trend/News Lifespan</a>

<a href='https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa#:~:text=Currently%2C%20the%20public%2Dfacing%20Google,of%20all%20the%20major%20candidates.'>Google Trends at Scale</a>

<a href='https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5119&rep=rep1&type=pdf'>FSM Predictors</a>

<a href='https://www.javatpoint.com/finite-state-machine'>FSM Intro</a>
