## Trend Forecasting using FSM






### Data
Google Trends provides Periodic Trends in the form of normalised values for keywords which are structured in a beautiful manner. Trends of multiple keywords can be compared and the relative trend data is obtained. These values are respectively scaled upto the one which has highest search volumes among a set of upto 5 keywords. Now, GT doesn't let us compare for more than a set of 5 keywords at once, hence we normalise it ourselves via mathematical means. Thanks to <a href='https://towardsdatascience.com/using-google-trends-at-scale-1c8b902b6bfa#:~:text=Currently%2C%20the%20public%2Dfacing%20Google,of%20all%20the%20major%20candidates.'>this</a>.
Every bit of the data was acquired from Google Trends via <a href='https://pypi.org/project/pytrends/'>PyTrends</a>, an unconventional api for GT.

## Data Preprocessing
Data is gathered by scraping from pytrends and later pre-processed followed by 
- Normalisation
- Concatenation 
- Minimization
- Moving averages (Moving Averages to focus and smooth up the trend slopes and directions) (_file)
- Transform (Take Transpose of every individual datapoint and make it more model specific) (_file)

## Local Maxima, Slopes and Trend Durations
Find the local peaks such that we obtain only the trending pattern, from the trend's initial ascend till its death which is proportional to a threshold which is peak specific.
Slopes calculated

