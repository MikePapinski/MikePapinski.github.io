---
layout: post
title: forex repeat.
summary: Check if it is possible to predict forex price movements only based on candlestick data.We will use 1h time-frame data set of EUR/USD during ~2014-2019 year.
featured-img: post_4
categories: [Sample, Guides]
---


## CHAPTER 5

<iframe src="https://giphy.com/embed/9u514UZd57mRhnBCEk" width="480" height="240" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/reaction-9u514UZd57mRhnBCEk">via GIPHY</a></p>

#### Let's leave the deep learning models for a while and try some simply statistics to create our strategy.

Is there any time during the week that the next candle will be most likely bullish or bearish?
Let's find out

##### Import Libraries


```python
import matplotlib
import numpy as np
import pandas as pd
import time
import datetime

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import seaborn as sns

print('Numpy version: ' + np.__version__)
print('Pandas version: ' + pd.__version__)
print('Matplotlib version: ' + matplotlib.__version__)
```

    Numpy version: 1.16.4
    Pandas version: 0.24.2
    Matplotlib version: 3.1.0



```python
float_data = pd.read_csv('Hour/EURUSD.csv')
print(len(float_data))
```

    101160



```python
float_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gmt time</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04.05.2003 21:00:00.000</td>
      <td>1.12284</td>
      <td>1.12338</td>
      <td>1.12242</td>
      <td>1.12305</td>
      <td>29059.0996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04.05.2003 22:00:00.000</td>
      <td>1.12274</td>
      <td>1.12302</td>
      <td>1.12226</td>
      <td>1.12241</td>
      <td>26091.8008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04.05.2003 23:00:00.000</td>
      <td>1.12235</td>
      <td>1.12235</td>
      <td>1.12160</td>
      <td>1.12169</td>
      <td>29240.9004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05.05.2003 00:00:00.000</td>
      <td>1.12161</td>
      <td>1.12314</td>
      <td>1.12154</td>
      <td>1.12258</td>
      <td>29914.8008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05.05.2003 01:00:00.000</td>
      <td>1.12232</td>
      <td>1.12262</td>
      <td>1.12099</td>
      <td>1.12140</td>
      <td>28370.6992</td>
    </tr>
    <tr>
      <th>5</th>
      <td>05.05.2003 02:00:00.000</td>
      <td>1.12141</td>
      <td>1.12211</td>
      <td>1.12085</td>
      <td>1.12152</td>
      <td>29867.6992</td>
    </tr>
    <tr>
      <th>6</th>
      <td>05.05.2003 03:00:00.000</td>
      <td>1.12123</td>
      <td>1.12179</td>
      <td>1.12049</td>
      <td>1.12162</td>
      <td>27256.8008</td>
    </tr>
    <tr>
      <th>7</th>
      <td>05.05.2003 04:00:00.000</td>
      <td>1.12098</td>
      <td>1.12176</td>
      <td>1.12079</td>
      <td>1.12122</td>
      <td>28278.9004</td>
    </tr>
    <tr>
      <th>8</th>
      <td>05.05.2003 05:00:00.000</td>
      <td>1.12129</td>
      <td>1.12222</td>
      <td>1.12091</td>
      <td>1.12143</td>
      <td>26764.9004</td>
    </tr>
    <tr>
      <th>9</th>
      <td>05.05.2003 06:00:00.000</td>
      <td>1.12128</td>
      <td>1.12241</td>
      <td>1.12066</td>
      <td>1.12088</td>
      <td>29956.5996</td>
    </tr>
  </tbody>
</table>
</div>




```python
def transform_to_heatmap(dataframe):
    dataframe.loc[dataframe['Close']>dataframe['Open'], 'color'] = 1
    dataframe.loc[dataframe['Close']<dataframe['Open'], 'color'] = -1
    del dataframe['Volume']
    del dataframe['Open']
    del dataframe['High']
    del dataframe['Low']
    del dataframe['Close']
    return dataframe
```


```python
float_data=transform_to_heatmap(float_data)
```


```python
float_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gmt time</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04.05.2003 21:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04.05.2003 22:00:00.000</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04.05.2003 23:00:00.000</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05.05.2003 00:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05.05.2003 01:00:00.000</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>05.05.2003 02:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>05.05.2003 03:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>05.05.2003 04:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>05.05.2003 05:00:00.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>05.05.2003 06:00:00.000</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_week_activity(raw_predictions,date_values,Filter=False):
    Monday_ls = [0] * 24
    Tuesday_ls = [0] * 24
    Wendsday_ls = [0] * 24
    Thursday_ls = [0] * 24
    Friday_ls = [0] * 24

    counter = 0

    for a in raw_predictions:
        raw_date_converted = datetime.datetime.strptime(date_values[counter], '%d.%m.%Y %H:%M:%S.000')
        week_day = raw_date_converted.weekday()
        hour = raw_date_converted.hour

        candle_check =[0,0]
        if a == 1: candle_check =[1,0]
        if a == -1: candle_check =[0,1]


        if week_day==0: Monday_ls[hour-1]=Monday_ls[hour-1]+np.array(candle_check)
        if week_day==1: Tuesday_ls[hour-1]=Tuesday_ls[hour-1]+np.array(candle_check)
        if week_day==2: Wendsday_ls[hour-1]=Wendsday_ls[hour-1]+np.array(candle_check)
        if week_day==3: Thursday_ls[hour-1]=Thursday_ls[hour-1]+np.array(candle_check)
        if week_day==4: Friday_ls[hour-1]=Friday_ls[hour-1]+np.array(candle_check)
        counter=counter+1


    final_list = [Monday_ls,Tuesday_ls,Wendsday_ls,Thursday_ls,Friday_ls]
    weekdays_count = 0
    for weekdays in final_list:
        hours_count = 0
        for hours in weekdays:
            if type(hours) != int:
                stats = round((hours[0]*100)/(hours[1]+hours[0]),0)
                if Filter==True:
                    if stats <= 55 and stats >= 45: stats=0
                    if stats < 45: stats= -stats

                final_list[weekdays_count][hours_count]=stats
            hours_count=hours_count+1
        weekdays_count=weekdays_count+1

    return [Monday_ls,Tuesday_ls,Wendsday_ls,Thursday_ls,Friday_ls]
```


```python

```


```python

```


```python
test=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
```


```python
def make_heat_map(dataset):


    weekdays=['Monday','Tuesday','Wendsday','Thursday','Friday']
    plt.figure(figsize=(18,5))
    ax = sns.heatmap(dataset,annot=True,annot_kws={"size": 10},
                        linewidth=0.5, yticklabels=weekdays,cmap="Blues")
    plt.title('Trade activity during the week')
    ax.set_ylabel('Week Days')
    ax.set_xlabel('Hours')

    plt.show
```

## EUR/USD 2003-2019 1h candles data


```python
make_heat_map(test)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_16_0.png)


## GBP/USD 2003-2019 1h candles data


```python
GBPUSD_dataset = pd.read_csv('Hour/GBPUSD.csv')
float_data=transform_to_heatmap(GBPUSD_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_18_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_18_1.png)


## USD/CAD 2003-2019 1h candles data


```python
USDCAD_dataset = pd.read_csv('Hour/USDCAD.csv')
float_data=transform_to_heatmap(USDCAD_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_20_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_20_1.png)


## NZD/USD 2003-2019 1h candles data


```python
NZDUSD_dataset = pd.read_csv('Hour/NZDUSD.csv')
float_data=transform_to_heatmap(NZDUSD_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_22_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_22_1.png)


## USD/JPY 2003-2019 1h candles data


```python
USDJPY_dataset = pd.read_csv('Hour/USDJPY.csv')
float_data=transform_to_heatmap(USDJPY_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_24_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_24_1.png)


## AUD/USD 2003-2019 1h candles data


```python
AUDUSD_dataset = pd.read_csv('Hour/AUDUSD.csv')
float_data=transform_to_heatmap(AUDUSD_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_26_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_26_1.png)


## USD/CHF 2003-2019 1h candles data


```python
USDCHF_dataset = pd.read_csv('Hour/USDCHF.csv')
float_data=transform_to_heatmap(USDCHF_dataset)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values)
make_heat_map(stats_list)
stats_list=get_week_activity(float_data['color'].values,float_data['Gmt time'].values,Filter=True)
make_heat_map(stats_list)
```


![png]({{ site.baseurl }}/assets/img/posts/post_4/output_28_0.png)



![png]({{ site.baseurl }}/assets/img/posts/post_4/output_28_1.png)



```python

```


```python

```
