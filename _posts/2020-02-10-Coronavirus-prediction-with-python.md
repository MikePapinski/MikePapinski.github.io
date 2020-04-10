---
layout: post
title: Coronavirus prediction with python and ML
author: MikePapinski
summary: This is a prediciton model to estimate how coronavirus outbreak will evolve over days.
categories: [Python]
image: assets/images/posts/9/post_9.JPG
---


# IMPORTS


```python
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
```


<p style="color: red;">
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
<a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>



# LOAD dataset


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
virusdata = "/content/drive/My Drive/data/2019_nCoV_data.csv"

```


```python
virus_data = pd.read_csv(virusdata)
```


```python
virus_data
```


<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sno</th>
      <th>Date</th>
      <th>Province/State</th>
      <th>Country</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>01/22/2020 12:00:00</td>
      <td>Anhui</td>
      <td>China</td>
      <td>2020-01-22 12:00:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>01/22/2020 12:00:00</td>
      <td>Beijing</td>
      <td>China</td>
      <td>2020-01-22 12:00:00</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>01/22/2020 12:00:00</td>
      <td>Chongqing</td>
      <td>China</td>
      <td>2020-01-22 12:00:00</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>01/22/2020 12:00:00</td>
      <td>Fujian</td>
      <td>China</td>
      <td>2020-01-22 12:00:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>01/22/2020 12:00:00</td>
      <td>Gansu</td>
      <td>China</td>
      <td>2020-01-22 12:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>695</th>
      <td>696</td>
      <td>02/03/2020 21:40:00</td>
      <td>Boston, MA</td>
      <td>US</td>
      <td>2020-01-02 19:43:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>696</th>
      <td>697</td>
      <td>02/03/2020 21:40:00</td>
      <td>Los Angeles, CA</td>
      <td>US</td>
      <td>2020-01-02 19:53:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>02/03/2020 21:40:00</td>
      <td>Orange, CA</td>
      <td>US</td>
      <td>2020-01-02 19:53:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>698</th>
      <td>699</td>
      <td>02/03/2020 21:40:00</td>
      <td>Seattle, WA</td>
      <td>US</td>
      <td>2020-01-02 19:43:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>699</th>
      <td>700</td>
      <td>02/03/2020 21:40:00</td>
      <td>Tempe, AZ</td>
      <td>US</td>
      <td>2020-01-02 19:43:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>700 rows × 8 columns</p>
</div>



# SUM ALL

### Confirmed infections


```python
import plotly.graph_objects as go
grouped_multiple = virus_data.groupby(['Date']).agg({'Confirmed': ['sum']})
grouped_multiple.columns = ['Confirmed ALL']
grouped_multiple = grouped_multiple.reset_index()
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Confirmed ALL'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='orange', width=2)))
fig.show()
```

