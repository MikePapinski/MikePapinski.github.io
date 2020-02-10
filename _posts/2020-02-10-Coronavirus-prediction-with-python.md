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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="097eae62-91d9-4251-8fed-618ed3ab4a92" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("097eae62-91d9-4251-8fed-618ed3ab4a92")) {
                    Plotly.newPlot(
                        '097eae62-91d9-4251-8fed-618ed3ab4a92',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "Deaths", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [555.0, 653.0, 941.0, 2019.0, 2794.0, 4473.0, 6057.0, 7783.0, 9776.0, 11374.0, 14549.0, 17295.0, 20588.0]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('097eae62-91d9-4251-8fed-618ed3ab4a92');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Europe vs China ALL


```python
china_vs_rest = virus_data.copy()
china_vs_rest.loc[china_vs_rest.Country == 'Mainland China', 'Country'] = "China"
china_vs_rest.loc[china_vs_rest.Country != 'China', 'Country'] = "Not China"
china_vs_rest = china_vs_rest.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})
china_vs_rest.columns = ['Confirmed ALL']
china_vs_rest = china_vs_rest.reset_index()
fig = px.line(china_vs_rest, x="Date", y="Confirmed ALL", color="Country",
              line_group="Country", hover_name="Country")
fig.update_layout(template='plotly_dark')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="f887214a-ba77-4e47-b309-f337f4d003e6" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("f887214a-ba77-4e47-b309-f337f4d003e6")) {
                    Plotly.newPlot(
                        'f887214a-ba77-4e47-b309-f337f4d003e6',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=China<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["China", "China", "China", "China", "China", "China", "China", "China", "China", "China", "China", "China", "China"], "legendgroup": "Country=China", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Country=China", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [549.0, 639.0, 916.0, 1979.0, 2737.0, 4409.0, 5970.0, 7678.0, 9658.0, 11221.0, 14375.0, 17114.0, 20400.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Not China<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China", "Not China"], "legendgroup": "Country=Not China", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Country=Not China", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [6.0, 14.0, 25.0, 40.0, 57.0, 64.0, 87.0, 105.0, 118.0, 153.0, 174.0, 181.0, 188.0], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Confirmed ALL"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('f887214a-ba77-4e47-b309-f337f4d003e6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Not China infections


```python
china_vs_rest = virus_data.copy()
china_vs_rest = china_vs_rest[china_vs_rest.Country != 'Mainland China']
china_vs_rest = china_vs_rest[china_vs_rest.Country != 'China']
china_vs_rest = china_vs_rest.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})
china_vs_rest.columns = ['Confirmed ALL']
china_vs_rest = china_vs_rest.reset_index()
fig = px.line(china_vs_rest, x="Date", y="Confirmed ALL", color="Country",
              line_group="Country", hover_name="Country")
fig.update_layout(template='plotly_dark')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="08c908ff-cc75-4cc2-b7cb-30d1918bbb7f" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("08c908ff-cc75-4cc2-b7cb-30d1918bbb7f")) {
                    Plotly.newPlot(
                        '08c908ff-cc75-4cc2-b7cb-30d1918bbb7f',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Japan<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan", "Japan"], "legendgroup": "Country=Japan", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Country=Japan", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 1.0, 2.0, 2.0, 4.0, 4.0, 7.0, 11.0, 11.0, 17.0, 20.0, 20.0, 20.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=South Korea<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea", "South Korea"], "legendgroup": "Country=South Korea", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Country=South Korea", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 6.0, 11.0, 15.0, 15.0, 15.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Thailand<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand"], "legendgroup": "Country=Thailand", "line": {"color": "#00cc96", "dash": "solid"}, "mode": "lines", "name": "Country=Thailand", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 3.0, 5.0, 7.0, 8.0, 8.0, 14.0, 14.0, 14.0, 19.0, 19.0, 19.0, 19.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=US<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["US", "US", "US", "US", "US", "US", "US", "US", "US", "US", "US", "US", "US"], "legendgroup": "Country=US", "line": {"color": "#ab63fa", "dash": "solid"}, "mode": "lines", "name": "Country=US", "showlegend": true, "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Australia<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Australia", "Australia", "Australia", "Australia", "Australia", "Australia", "Australia", "Australia", "Australia", "Australia", "Australia"], "legendgroup": "Country=Australia", "line": {"color": "#FFA15A", "dash": "solid"}, "mode": "lines", "name": "Country=Australia", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [0.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, 9.0, 12.0, 12.0, 12.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Brazil<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Brazil"], "legendgroup": "Country=Brazil", "line": {"color": "#19d3f3", "dash": "solid"}, "mode": "lines", "name": "Country=Brazil", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00"], "xaxis": "x", "y": [0.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Hong Kong<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong", "Hong Kong"], "legendgroup": "Country=Hong Kong", "line": {"color": "#FF6692", "dash": "solid"}, "mode": "lines", "name": "Country=Hong Kong", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 5.0, 8.0, 8.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 15.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Macau<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau", "Macau"], "legendgroup": "Country=Macau", "line": {"color": "#B6E880", "dash": "solid"}, "mode": "lines", "name": "Country=Macau", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 2.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Malaysia<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia"], "legendgroup": "Country=Malaysia", "line": {"color": "#FF97FF", "dash": "solid"}, "mode": "lines", "name": "Country=Malaysia", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [0.0, 3.0, 4.0, 4.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Mexico<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Mexico"], "legendgroup": "Country=Mexico", "line": {"color": "#FECB52", "dash": "solid"}, "mode": "lines", "name": "Country=Mexico", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00"], "xaxis": "x", "y": [0.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Philippines<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Philippines", "Philippines", "Philippines", "Philippines", "Philippines", "Philippines"], "legendgroup": "Country=Philippines", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Country=Philippines", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [0.0, 1.0, 1.0, 2.0, 2.0, 2.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Singapore<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore"], "legendgroup": "Country=Singapore", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Country=Singapore", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 3.0, 3.0, 4.0, 5.0, 7.0, 10.0, 10.0, 16.0, 18.0, 18.0, 18.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Taiwan<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan", "Taiwan"], "legendgroup": "Country=Taiwan", "line": {"color": "#00cc96", "dash": "solid"}, "mode": "lines", "name": "Country=Taiwan", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 3.0, 3.0, 4.0, 5.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 10.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Vietnam<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam", "Vietnam"], "legendgroup": "Country=Vietnam", "line": {"color": "#ab63fa", "dash": "solid"}, "mode": "lines", "name": "Country=Vietnam", "showlegend": true, "type": "scatter", "x": ["01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 8.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=France<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["France", "France", "France", "France", "France", "France", "France", "France", "France", "France", "France"], "legendgroup": "Country=France", "line": {"color": "#FFA15A", "dash": "solid"}, "mode": "lines", "name": "Country=France", "showlegend": true, "type": "scatter", "x": ["01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Nepal<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Nepal", "Nepal", "Nepal", "Nepal", "Nepal", "Nepal", "Nepal", "Nepal", "Nepal", "Nepal"], "legendgroup": "Country=Nepal", "line": {"color": "#19d3f3", "dash": "solid"}, "mode": "lines", "name": "Country=Nepal", "showlegend": true, "type": "scatter", "x": ["01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Canada<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Canada", "Canada", "Canada", "Canada", "Canada", "Canada", "Canada", "Canada", "Canada"], "legendgroup": "Country=Canada", "line": {"color": "#FF6692", "dash": "solid"}, "mode": "lines", "name": "Country=Canada", "showlegend": true, "type": "scatter", "x": ["01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Cambodia<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia"], "legendgroup": "Country=Cambodia", "line": {"color": "#B6E880", "dash": "solid"}, "mode": "lines", "name": "Country=Cambodia", "showlegend": true, "type": "scatter", "x": ["01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Germany<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Germany", "Germany", "Germany", "Germany", "Germany", "Germany", "Germany", "Germany"], "legendgroup": "Country=Germany", "line": {"color": "#FF97FF", "dash": "solid"}, "mode": "lines", "name": "Country=Germany", "showlegend": true, "type": "scatter", "x": ["01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 4.0, 4.0, 4.0, 7.0, 8.0, 10.0, 12.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Ivory Coast<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Ivory Coast"], "legendgroup": "Country=Ivory Coast", "line": {"color": "#FECB52", "dash": "solid"}, "mode": "lines", "name": "Country=Ivory Coast", "showlegend": true, "type": "scatter", "x": ["01/27/2020 20:30:00"], "xaxis": "x", "y": [0.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Sri Lanka<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Sri Lanka", "Sri Lanka", "Sri Lanka", "Sri Lanka", "Sri Lanka", "Sri Lanka", "Sri Lanka", "Sri Lanka"], "legendgroup": "Country=Sri Lanka", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Country=Sri Lanka", "showlegend": true, "type": "scatter", "x": ["01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Finland<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Finland", "Finland", "Finland", "Finland", "Finland", "Finland"], "legendgroup": "Country=Finland", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Country=Finland", "showlegend": true, "type": "scatter", "x": ["01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=United Arab Emirates<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["United Arab Emirates", "United Arab Emirates", "United Arab Emirates", "United Arab Emirates", "United Arab Emirates", "United Arab Emirates"], "legendgroup": "Country=United Arab Emirates", "line": {"color": "#00cc96", "dash": "solid"}, "mode": "lines", "name": "Country=United Arab Emirates", "showlegend": true, "type": "scatter", "x": ["01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [4.0, 4.0, 4.0, 4.0, 5.0, 5.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=India<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["India", "India", "India", "India", "India"], "legendgroup": "Country=India", "line": {"color": "#ab63fa", "dash": "solid"}, "mode": "lines", "name": "Country=India", "showlegend": true, "type": "scatter", "x": ["01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 2.0, 3.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Italy<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Italy", "Italy", "Italy", "Italy", "Italy"], "legendgroup": "Country=Italy", "line": {"color": "#FFA15A", "dash": "solid"}, "mode": "lines", "name": "Country=Italy", "showlegend": true, "type": "scatter", "x": ["01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 2.0, 2.0, 2.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Russia<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Russia", "Russia", "Russia", "Russia"], "legendgroup": "Country=Russia", "line": {"color": "#19d3f3", "dash": "solid"}, "mode": "lines", "name": "Country=Russia", "showlegend": true, "type": "scatter", "x": ["01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 2.0, 2.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Spain<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Spain", "Spain", "Spain", "Spain"], "legendgroup": "Country=Spain", "line": {"color": "#FF6692", "dash": "solid"}, "mode": "lines", "name": "Country=Spain", "showlegend": true, "type": "scatter", "x": ["01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=Sweden<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["Sweden", "Sweden", "Sweden", "Sweden"], "legendgroup": "Country=Sweden", "line": {"color": "#B6E880", "dash": "solid"}, "mode": "lines", "name": "Country=Sweden", "showlegend": true, "type": "scatter", "x": ["01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [1.0, 1.0, 1.0, 1.0], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Country=UK<br>Date=%{x}<br>Confirmed ALL=%{y}", "hovertext": ["UK", "UK", "UK", "UK"], "legendgroup": "Country=UK", "line": {"color": "#FF97FF", "dash": "solid"}, "mode": "lines", "name": "Country=UK", "showlegend": true, "type": "scatter", "x": ["01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "xaxis": "x", "y": [2.0, 2.0, 2.0, 2.0], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Confirmed ALL"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('08c908ff-cc75-4cc2-b7cb-30d1918bbb7f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


### Deaths and healings


```python
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum']})
grouped_multiple.columns = ['Deaths ALL','Recovered ALL']
grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Deaths ALL'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Recovered ALL'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='green', width=2)))
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="d3688b48-73a5-4a5d-8e0c-67cd2a15283b" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("d3688b48-73a5-4a5d-8e0c-67cd2a15283b")) {
                    Plotly.newPlot(
                        'd3688b48-73a5-4a5d-8e0c-67cd2a15283b',
                        [{"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "Deaths", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 18.0, 26.0, 56.0, 80.0, 107.0, 132.0, 170.0, 213.0, 259.0, 305.0, 362.0, 426.0]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "Recovered", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 30.0, 36.0, 49.0, 54.0, 63.0, 110.0, 133.0, 187.0, 252.0, 340.0, 487.0, 644.0]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('d3688b48-73a5-4a5d-8e0c-67cd2a15283b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# % rate deaths to recovery


```python
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum']})
grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL']
grouped_multiple = grouped_multiple.reset_index()

grouped_multiple['Deaths_ALL_%'] = grouped_multiple.apply(lambda row: (row.Deaths_ALL*100)//
                                               (row.Deaths_ALL + row.Recovered_ALL) 
                                               if row.Deaths_ALL  else 0, axis=1)

grouped_multiple['Recovered_ALL_%'] = grouped_multiple.apply(lambda row: (row.Recovered_ALL*100)//
                                               (row.Deaths_ALL + row.Recovered_ALL) 
                                               if row.Deaths_ALL  else 0, axis=1)


fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Deaths_ALL_%'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Recovered_ALL_%'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='green', width=2)))
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="aa6569e8-94a7-43ed-9046-d96853bf21fb" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("aa6569e8-94a7-43ed-9046-d96853bf21fb")) {
                    Plotly.newPlot(
                        'aa6569e8-94a7-43ed-9046-d96853bf21fb',
                        [{"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "Deaths", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 37.0, 41.0, 53.0, 59.0, 62.0, 54.0, 56.0, 53.0, 50.0, 47.0, 42.0, 39.0]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "Recovered", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 62.0, 58.0, 46.0, 40.0, 37.0, 45.0, 43.0, 46.0, 49.0, 52.0, 57.0, 60.0]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('aa6569e8-94a7-43ed-9046-d96853bf21fb');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# What are we going to predict?

### All infections vs (recovery + deaths)


```python
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum'], 'Confirmed': ['sum']})
grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL', 'All']
grouped_multiple = grouped_multiple.reset_index()
grouped_multiple['Deaths_Revocered'] = grouped_multiple.apply(lambda row: row.Deaths_ALL + row.Recovered_ALL, axis=1)

fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Deaths_ALL'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Recovered_ALL'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['All'],
                         mode='lines+markers',
                         name='All',
                         line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 
                         y=grouped_multiple['Deaths_Revocered'],
                         mode='lines+markers',
                         name='Deaths + Recovered',
                         line=dict(color='white', width=2)))

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="2bd08fcd-ca2a-4cbd-bcbf-b03dacf332ea" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("2bd08fcd-ca2a-4cbd-bcbf-b03dacf332ea")) {
                    Plotly.newPlot(
                        '2bd08fcd-ca2a-4cbd-bcbf-b03dacf332ea',
                        [{"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "Deaths", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 18.0, 26.0, 56.0, 80.0, 107.0, 132.0, 170.0, 213.0, 259.0, 305.0, 362.0, 426.0]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "Recovered", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 30.0, 36.0, 49.0, 54.0, 63.0, 110.0, 133.0, 187.0, 252.0, 340.0, 487.0, 644.0]}, {"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "All", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [555.0, 653.0, 941.0, 2019.0, 2794.0, 4473.0, 6057.0, 7783.0, 9776.0, 11374.0, 14549.0, 17295.0, 20588.0]}, {"line": {"color": "white", "width": 2}, "mode": "lines+markers", "name": "Deaths + Recovered", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 48.0, 62.0, 105.0, 134.0, 170.0, 242.0, 303.0, 400.0, 511.0, 645.0, 849.0, 1070.0]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('2bd08fcd-ca2a-4cbd-bcbf-b03dacf332ea');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Calculate % Returns


```python
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum'], 'Confirmed': ['sum']})
grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL', 'All']
grouped_multiple = grouped_multiple.reset_index()
grouped_multiple['Deaths_Revocered'] = grouped_multiple.apply(lambda row: row.Deaths_ALL + row.Recovered_ALL, axis=1)
grouped_multiple
```




<div>

<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Deaths_ALL</th>
      <th>Recovered_ALL</th>
      <th>All</th>
      <th>Deaths_Revocered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/22/2020 12:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>555.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/23/2020 12:00:00</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>653.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01/24/2020 12:00:00</td>
      <td>26.0</td>
      <td>36.0</td>
      <td>941.0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01/25/2020 22:00:00</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>2019.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01/26/2020 23:00:00</td>
      <td>80.0</td>
      <td>54.0</td>
      <td>2794.0</td>
      <td>134.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01/27/2020 20:30:00</td>
      <td>107.0</td>
      <td>63.0</td>
      <td>4473.0</td>
      <td>170.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01/28/2020 23:00:00</td>
      <td>132.0</td>
      <td>110.0</td>
      <td>6057.0</td>
      <td>242.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>01/29/2020 21:00:00</td>
      <td>170.0</td>
      <td>133.0</td>
      <td>7783.0</td>
      <td>303.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>01/30/2020 21:30:00</td>
      <td>213.0</td>
      <td>187.0</td>
      <td>9776.0</td>
      <td>400.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>01/31/2020 19:00:00</td>
      <td>259.0</td>
      <td>252.0</td>
      <td>11374.0</td>
      <td>511.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>02/01/2020 23:00:00</td>
      <td>305.0</td>
      <td>340.0</td>
      <td>14549.0</td>
      <td>645.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>02/02/2020 21:00:00</td>
      <td>362.0</td>
      <td>487.0</td>
      <td>17295.0</td>
      <td>849.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>02/03/2020 21:40:00</td>
      <td>426.0</td>
      <td>644.0</td>
      <td>20588.0</td>
      <td>1070.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_multiple['infections_perc'] = grouped_multiple['All'].pct_change(1)
grouped_multiple['recovered_perc'] = grouped_multiple['Recovered_ALL'].pct_change(1)
grouped_multiple['death_perc'] = grouped_multiple['Deaths_ALL'].pct_change(1)
grouped_multiple = grouped_multiple.replace([np.inf, -np.inf], np.nan)
main_df=grouped_multiple.fillna(0)
main_df
```




<div>

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Deaths_ALL</th>
      <th>Recovered_ALL</th>
      <th>All</th>
      <th>Deaths_Revocered</th>
      <th>infections_perc</th>
      <th>recovered_perc</th>
      <th>death_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/22/2020 12:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>555.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/23/2020 12:00:00</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>653.0</td>
      <td>48.0</td>
      <td>0.176577</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01/24/2020 12:00:00</td>
      <td>26.0</td>
      <td>36.0</td>
      <td>941.0</td>
      <td>62.0</td>
      <td>0.441041</td>
      <td>0.200000</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01/25/2020 22:00:00</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>2019.0</td>
      <td>105.0</td>
      <td>1.145590</td>
      <td>0.361111</td>
      <td>1.153846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01/26/2020 23:00:00</td>
      <td>80.0</td>
      <td>54.0</td>
      <td>2794.0</td>
      <td>134.0</td>
      <td>0.383853</td>
      <td>0.102041</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01/27/2020 20:30:00</td>
      <td>107.0</td>
      <td>63.0</td>
      <td>4473.0</td>
      <td>170.0</td>
      <td>0.600931</td>
      <td>0.166667</td>
      <td>0.337500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01/28/2020 23:00:00</td>
      <td>132.0</td>
      <td>110.0</td>
      <td>6057.0</td>
      <td>242.0</td>
      <td>0.354125</td>
      <td>0.746032</td>
      <td>0.233645</td>
    </tr>
    <tr>
      <th>7</th>
      <td>01/29/2020 21:00:00</td>
      <td>170.0</td>
      <td>133.0</td>
      <td>7783.0</td>
      <td>303.0</td>
      <td>0.284960</td>
      <td>0.209091</td>
      <td>0.287879</td>
    </tr>
    <tr>
      <th>8</th>
      <td>01/30/2020 21:30:00</td>
      <td>213.0</td>
      <td>187.0</td>
      <td>9776.0</td>
      <td>400.0</td>
      <td>0.256071</td>
      <td>0.406015</td>
      <td>0.252941</td>
    </tr>
    <tr>
      <th>9</th>
      <td>01/31/2020 19:00:00</td>
      <td>259.0</td>
      <td>252.0</td>
      <td>11374.0</td>
      <td>511.0</td>
      <td>0.163462</td>
      <td>0.347594</td>
      <td>0.215962</td>
    </tr>
    <tr>
      <th>10</th>
      <td>02/01/2020 23:00:00</td>
      <td>305.0</td>
      <td>340.0</td>
      <td>14549.0</td>
      <td>645.0</td>
      <td>0.279145</td>
      <td>0.349206</td>
      <td>0.177606</td>
    </tr>
    <tr>
      <th>11</th>
      <td>02/02/2020 21:00:00</td>
      <td>362.0</td>
      <td>487.0</td>
      <td>17295.0</td>
      <td>849.0</td>
      <td>0.188741</td>
      <td>0.432353</td>
      <td>0.186885</td>
    </tr>
    <tr>
      <th>12</th>
      <td>02/03/2020 21:40:00</td>
      <td>426.0</td>
      <td>644.0</td>
      <td>20588.0</td>
      <td>1070.0</td>
      <td>0.190402</td>
      <td>0.322382</td>
      <td>0.176796</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=main_df['Date'], 
                         y=main_df['infections_perc'],
                         mode='lines+markers',
                         name='infections_perc',
                         line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=main_df['Date'], 
                         y=main_df['recovered_perc'],
                         mode='lines+markers',
                         name='recovered_perc',
                         line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=main_df['Date'], 
                         y=main_df['death_perc'],
                         mode='lines+markers',
                         name='death_perc',
                         line=dict(color='red', width=2)))
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="4bb84c90-23d6-4d85-9a82-45cf0d779849" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("4bb84c90-23d6-4d85-9a82-45cf0d779849")) {
                    Plotly.newPlot(
                        '4bb84c90-23d6-4d85-9a82-45cf0d779849',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "infections_perc", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 0.17657657657657655, 0.44104134762633995, 1.1455897980871415, 0.38385339276869734, 0.6009305654974946, 0.3541247484909458, 0.2849595509328051, 0.2560709238083001, 0.16346153846153855, 0.2791454193775278, 0.1887414942607739, 0.19040185024573586]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "recovered_perc", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 0.0, 0.19999999999999996, 0.36111111111111116, 0.1020408163265305, 0.16666666666666674, 0.746031746031746, 0.209090909090909, 0.40601503759398505, 0.3475935828877006, 0.3492063492063493, 0.4323529411764706, 0.322381930184805]}, {"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "death_perc", "type": "scatter", "x": ["01/22/2020 12:00:00", "01/23/2020 12:00:00", "01/24/2020 12:00:00", "01/25/2020 22:00:00", "01/26/2020 23:00:00", "01/27/2020 20:30:00", "01/28/2020 23:00:00", "01/29/2020 21:00:00", "01/30/2020 21:30:00", "01/31/2020 19:00:00", "02/01/2020 23:00:00", "02/02/2020 21:00:00", "02/03/2020 21:40:00"], "y": [0.0, 0.0, 0.4444444444444444, 1.1538461538461537, 0.4285714285714286, 0.3374999999999999, 0.23364485981308403, 0.28787878787878785, 0.2529411764705882, 0.215962441314554, 0.1776061776061777, 0.18688524590163924, 0.17679558011049723]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('4bb84c90-23d6-4d85-9a82-45cf0d779849');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Not enough data for model training!
### Split this data for more samples


```python
def IncreaseData(dflist):
  NewList=[]
  add_this=0
  for split_value in dflist:
      increment_by = int((split_value-add_this)//24)
      for new_val in range(24):
          add_this=increment_by+add_this
          NewList.append(add_this)
  return NewList

inc_total = IncreaseData(grouped_multiple['All'])
inc_death = IncreaseData(grouped_multiple['Deaths_ALL'])
inc_rec = IncreaseData(grouped_multiple['Recovered_ALL'])
df = pd.DataFrame(list(zip(inc_total, inc_death, inc_rec)), 
               columns =['inc_total', 'inc_death', 'inc_rec']) 
df
```




<div>

<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inc_total</th>
      <th>inc_death</th>
      <th>inc_rec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>115</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307</th>
      <td>20020</td>
      <td>400</td>
      <td>600</td>
    </tr>
    <tr>
      <th>308</th>
      <td>20157</td>
      <td>402</td>
      <td>606</td>
    </tr>
    <tr>
      <th>309</th>
      <td>20294</td>
      <td>404</td>
      <td>612</td>
    </tr>
    <tr>
      <th>310</th>
      <td>20431</td>
      <td>406</td>
      <td>618</td>
    </tr>
    <tr>
      <th>311</th>
      <td>20568</td>
      <td>408</td>
      <td>624</td>
    </tr>
  </tbody>
</table>
<p>312 rows × 3 columns</p>
</div>




```python
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=df.index, 
                         y=df['inc_total'],
                         mode='lines',
                         name='inc_total',
                         line=dict(color='orange', width=2)))
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=df.index, 
                         y=df['inc_rec'],
                         mode='lines',
                         name='inc_rec',
                         line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=df.index, 
                         y=df['inc_death'],
                         mode='lines',
                         name='inc_death',
                         line=dict(color='red', width=2)))

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="6f6c3d0d-3f47-4e83-938c-c10ac2ca48c6" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("6f6c3d0d-3f47-4e83-938c-c10ac2ca48c6")) {
                    Plotly.newPlot(
                        '6f6c3d0d-3f47-4e83-938c-c10ac2ca48c6',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines", "name": "inc_total", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311], "y": [23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 529, 552, 556, 560, 564, 568, 572, 576, 580, 584, 588, 592, 596, 600, 604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 981, 1026, 1071, 1116, 1161, 1206, 1251, 1296, 1341, 1386, 1431, 1476, 1521, 1566, 1611, 1656, 1701, 1746, 1791, 1836, 1881, 1926, 1971, 2016, 2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272, 2304, 2336, 2368, 2400, 2432, 2464, 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720, 2752, 2784, 2854, 2924, 2994, 3064, 3134, 3204, 3274, 3344, 3414, 3484, 3554, 3624, 3694, 3764, 3834, 3904, 3974, 4044, 4114, 4184, 4254, 4324, 4394, 4464, 4530, 4596, 4662, 4728, 4794, 4860, 4926, 4992, 5058, 5124, 5190, 5256, 5322, 5388, 5454, 5520, 5586, 5652, 5718, 5784, 5850, 5916, 5982, 6048, 6120, 6192, 6264, 6336, 6408, 6480, 6552, 6624, 6696, 6768, 6840, 6912, 6984, 7056, 7128, 7200, 7272, 7344, 7416, 7488, 7560, 7632, 7704, 7776, 7859, 7942, 8025, 8108, 8191, 8274, 8357, 8440, 8523, 8606, 8689, 8772, 8855, 8938, 9021, 9104, 9187, 9270, 9353, 9436, 9519, 9602, 9685, 9768, 9834, 9900, 9966, 10032, 10098, 10164, 10230, 10296, 10362, 10428, 10494, 10560, 10626, 10692, 10758, 10824, 10890, 10956, 11022, 11088, 11154, 11220, 11286, 11352, 11485, 11618, 11751, 11884, 12017, 12150, 12283, 12416, 12549, 12682, 12815, 12948, 13081, 13214, 13347, 13480, 13613, 13746, 13879, 14012, 14145, 14278, 14411, 14544, 14658, 14772, 14886, 15000, 15114, 15228, 15342, 15456, 15570, 15684, 15798, 15912, 16026, 16140, 16254, 16368, 16482, 16596, 16710, 16824, 16938, 17052, 17166, 17280, 17417, 17554, 17691, 17828, 17965, 18102, 18239, 18376, 18513, 18650, 18787, 18924, 19061, 19198, 19335, 19472, 19609, 19746, 19883, 20020, 20157, 20294, 20431, 20568]}, {"line": {"color": "green", "width": 2}, "mode": "lines", "name": "inc_rec", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426, 432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498, 504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594, 600, 606, 612, 618, 624]}, {"line": {"color": "red", "width": 2}, "mode": "lines", "name": "inc_death", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('6f6c3d0d-3f47-4e83-938c-c10ac2ca48c6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
inc_total_perc = df['inc_total'].pct_change(1)
inc_death_perc = df['inc_death'].pct_change(1)
inc_rec_perc = df['inc_rec'].pct_change(1)
dff = pd.DataFrame(list(zip(inc_total_perc, inc_death_perc, inc_rec_perc)), 
              columns =['inc_total_perc', 'inc_death_perc', 'inc_rec_perc']) 
dff=dff.replace([np.inf, -np.inf], np.nan)
dff=dff.fillna(0)
dff
```




<div>

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inc_total_perc</th>
      <th>inc_death_perc</th>
      <th>inc_rec_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>307</th>
      <td>0.006890</td>
      <td>0.005025</td>
      <td>0.010101</td>
    </tr>
    <tr>
      <th>308</th>
      <td>0.006843</td>
      <td>0.005000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>309</th>
      <td>0.006797</td>
      <td>0.004975</td>
      <td>0.009901</td>
    </tr>
    <tr>
      <th>310</th>
      <td>0.006751</td>
      <td>0.004950</td>
      <td>0.009804</td>
    </tr>
    <tr>
      <th>311</th>
      <td>0.006705</td>
      <td>0.004926</td>
      <td>0.009709</td>
    </tr>
  </tbody>
</table>
<p>312 rows × 3 columns</p>
</div>



# Class for LSTM training


```python
class TrainLSTM():
  def create_dataset(self, dataset, look_back=1, column = 0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
      a = dataset[i:(i+look_back), column]
      dataX.append(a)
      dataY.append(dataset[i + look_back, column])
    return np.array(dataX), np.array(dataY)
  
  def TrainModel(self, dframe, column):
    df = dframe.values
    df = df.astype('float32')
    train_size = int(len(df) * 0.90)
    test_size = len(df) - train_size
    Train, Validate = df[0:train_size,:], df[train_size:len(df),:]
    look_back = 24
    trainX, trainY = self.create_dataset(Train, look_back, column)
    testX, testY = self.create_dataset(Validate, look_back, column)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    self.trainPredict = model.predict(trainX)
    self.testPredict = model.predict(testX)
    trainScore = math.sqrt(mean_squared_error(trainY, self.trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, self.testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    Model_Prediciton_Resolved=[]
    lastDT=testX[0][0]
    print(lastDT)
    for i in range(168):
        predi = model.predict(np.array([[lastDT]]))
        Model_Prediciton_Resolved.append(predi[0][0])
        lastDT = lastDT[:-1]
        lastDT = np.append(predi, lastDT)
       
    return Model_Prediciton_Resolved
```

# Train model for total infections prediction


```python
NeuralNets = TrainLSTM()
result_total = NeuralNets.TrainModel(dff,0)
result_death = NeuralNets.TrainModel(dff,1)
result_rec = NeuralNets.TrainModel(dff,2)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    Train on 255 samples
    Epoch 1/100
    255/255 - 1s - loss: 1.4664e-04
    Epoch 2/100
    255/255 - 1s - loss: 9.6173e-05
    Epoch 3/100
    255/255 - 1s - loss: 6.1573e-05
    Epoch 4/100
    255/255 - 1s - loss: 5.1483e-05
    Epoch 5/100
    255/255 - 1s - loss: 3.8186e-05
    Epoch 6/100
    255/255 - 1s - loss: 2.8144e-05
    Epoch 7/100
    255/255 - 1s - loss: 2.3017e-05
    Epoch 8/100
    255/255 - 1s - loss: 2.0399e-05
    Epoch 9/100
    255/255 - 1s - loss: 1.6852e-05
    Epoch 10/100
    255/255 - 1s - loss: 1.6612e-05
    Epoch 11/100
    255/255 - 1s - loss: 1.4900e-05
    Epoch 12/100
    255/255 - 1s - loss: 1.4541e-05
    Epoch 13/100
    255/255 - 1s - loss: 1.3714e-05
    Epoch 14/100
    255/255 - 1s - loss: 1.4570e-05
    Epoch 15/100
    255/255 - 1s - loss: 1.2805e-05
    Epoch 16/100
    255/255 - 1s - loss: 1.3365e-05
    Epoch 17/100
    255/255 - 1s - loss: 1.2795e-05
    Epoch 18/100
    255/255 - 1s - loss: 1.2425e-05
    Epoch 19/100
    255/255 - 0s - loss: 1.2523e-05
    Epoch 20/100
    255/255 - 0s - loss: 1.1558e-05
    Epoch 21/100
    255/255 - 0s - loss: 1.4989e-05
    Epoch 22/100
    255/255 - 0s - loss: 1.2397e-05
    Epoch 23/100
    255/255 - 0s - loss: 1.2345e-05
    Epoch 24/100
    255/255 - 0s - loss: 1.0672e-05
    Epoch 25/100
    255/255 - 0s - loss: 1.6244e-05
    Epoch 26/100
    255/255 - 1s - loss: 1.2589e-05
    Epoch 27/100
    255/255 - 0s - loss: 1.1183e-05
    Epoch 28/100
    255/255 - 0s - loss: 1.1890e-05
    Epoch 29/100
    255/255 - 1s - loss: 1.1047e-05
    Epoch 30/100
    255/255 - 1s - loss: 1.0532e-05
    Epoch 31/100
    255/255 - 1s - loss: 1.4254e-05
    Epoch 32/100
    255/255 - 1s - loss: 1.0738e-05
    Epoch 33/100
    255/255 - 1s - loss: 1.0376e-05
    Epoch 34/100
    255/255 - 1s - loss: 1.1442e-05
    Epoch 35/100
    255/255 - 1s - loss: 9.9785e-06
    Epoch 36/100
    255/255 - 1s - loss: 1.1246e-05
    Epoch 37/100
    255/255 - 1s - loss: 9.6406e-06
    Epoch 38/100
    255/255 - 1s - loss: 9.9646e-06
    Epoch 39/100
    255/255 - 1s - loss: 1.1146e-05
    Epoch 40/100
    255/255 - 1s - loss: 1.2285e-05
    Epoch 41/100
    255/255 - 1s - loss: 1.0037e-05
    Epoch 42/100
    255/255 - 1s - loss: 1.0134e-05
    Epoch 43/100
    255/255 - 1s - loss: 1.1594e-05
    Epoch 44/100
    255/255 - 1s - loss: 1.1222e-05
    Epoch 45/100
    255/255 - 1s - loss: 1.0040e-05
    Epoch 46/100
    255/255 - 1s - loss: 9.7788e-06
    Epoch 47/100
    255/255 - 1s - loss: 1.0057e-05
    Epoch 48/100
    255/255 - 0s - loss: 1.0675e-05
    Epoch 49/100
    255/255 - 1s - loss: 1.0098e-05
    Epoch 50/100
    255/255 - 1s - loss: 1.0173e-05
    Epoch 51/100
    255/255 - 1s - loss: 1.0066e-05
    Epoch 52/100
    255/255 - 1s - loss: 1.0225e-05
    Epoch 53/100
    255/255 - 1s - loss: 9.8271e-06
    Epoch 54/100
    255/255 - 0s - loss: 9.1116e-06
    Epoch 55/100
    255/255 - 0s - loss: 9.7831e-06
    Epoch 56/100
    255/255 - 1s - loss: 9.2709e-06
    Epoch 57/100
    255/255 - 0s - loss: 9.7566e-06
    Epoch 58/100
    255/255 - 0s - loss: 9.4916e-06
    Epoch 59/100
    255/255 - 0s - loss: 1.1668e-05
    Epoch 60/100
    255/255 - 0s - loss: 9.4166e-06
    Epoch 61/100
    255/255 - 0s - loss: 9.9400e-06
    Epoch 62/100
    255/255 - 0s - loss: 1.0626e-05
    Epoch 63/100
    255/255 - 1s - loss: 1.0031e-05
    Epoch 64/100
    255/255 - 1s - loss: 9.8724e-06
    Epoch 65/100
    255/255 - 1s - loss: 1.0964e-05
    Epoch 66/100
    255/255 - 1s - loss: 9.6836e-06
    Epoch 67/100
    255/255 - 1s - loss: 9.4829e-06
    Epoch 68/100
    255/255 - 1s - loss: 8.8323e-06
    Epoch 69/100
    255/255 - 1s - loss: 1.0630e-05
    Epoch 70/100
    255/255 - 0s - loss: 9.4087e-06
    Epoch 71/100
    255/255 - 1s - loss: 9.1316e-06
    Epoch 72/100
    255/255 - 1s - loss: 1.0854e-05
    Epoch 73/100
    255/255 - 0s - loss: 9.8154e-06
    Epoch 74/100
    255/255 - 0s - loss: 9.4421e-06
    Epoch 75/100
    255/255 - 1s - loss: 9.0475e-06
    Epoch 76/100
    255/255 - 1s - loss: 8.8974e-06
    Epoch 77/100
    255/255 - 1s - loss: 8.9514e-06
    Epoch 78/100
    255/255 - 1s - loss: 8.8892e-06
    Epoch 79/100
    255/255 - 1s - loss: 8.7603e-06
    Epoch 80/100
    255/255 - 1s - loss: 9.9406e-06
    Epoch 81/100
    255/255 - 0s - loss: 9.0235e-06
    Epoch 82/100
    255/255 - 1s - loss: 9.4853e-06
    Epoch 83/100
    255/255 - 0s - loss: 9.2856e-06
    Epoch 84/100
    255/255 - 1s - loss: 9.4011e-06
    Epoch 85/100
    255/255 - 1s - loss: 8.9840e-06
    Epoch 86/100
    255/255 - 1s - loss: 1.0039e-05
    Epoch 87/100
    255/255 - 1s - loss: 9.0924e-06
    Epoch 88/100
    255/255 - 1s - loss: 9.4557e-06
    Epoch 89/100
    255/255 - 1s - loss: 9.3530e-06
    Epoch 90/100
    255/255 - 1s - loss: 8.8410e-06
    Epoch 91/100
    255/255 - 1s - loss: 8.7258e-06
    Epoch 92/100
    255/255 - 1s - loss: 1.0180e-05
    Epoch 93/100
    255/255 - 1s - loss: 9.3253e-06
    Epoch 94/100
    255/255 - 1s - loss: 8.7115e-06
    Epoch 95/100
    255/255 - 1s - loss: 8.7225e-06
    Epoch 96/100
    255/255 - 1s - loss: 8.9636e-06
    Epoch 97/100
    255/255 - 1s - loss: 9.8459e-06
    Epoch 98/100
    255/255 - 1s - loss: 8.9508e-06
    Epoch 99/100
    255/255 - 1s - loss: 8.9070e-06
    Epoch 100/100
    255/255 - 1s - loss: 8.9382e-06
    Train Score: 0.00 RMSE
    Test Score: 0.00 RMSE
    [0.00696481 0.00691664 0.00686913 0.00682226 0.00677603 0.00673043
     0.00668543 0.00664103 0.00792824 0.00786588 0.00780449 0.00774405
     0.00768454 0.00762594 0.00756822 0.00751138 0.00745538 0.00740021
     0.00734584 0.00729228 0.00723948 0.00718745 0.00713616 0.0070856 ]
    Train on 255 samples
    Epoch 1/100
    255/255 - 1s - loss: 0.0058
    Epoch 2/100
    255/255 - 1s - loss: 0.0055
    Epoch 3/100
    255/255 - 1s - loss: 0.0054
    Epoch 4/100
    255/255 - 1s - loss: 0.0053
    Epoch 5/100
    255/255 - 1s - loss: 0.0051
    Epoch 6/100
    255/255 - 1s - loss: 0.0050
    Epoch 7/100
    255/255 - 1s - loss: 0.0049
    Epoch 8/100
    255/255 - 1s - loss: 0.0048
    Epoch 9/100
    255/255 - 1s - loss: 0.0048
    Epoch 10/100
    255/255 - 1s - loss: 0.0047
    Epoch 11/100
    255/255 - 1s - loss: 0.0046
    Epoch 12/100
    255/255 - 1s - loss: 0.0046
    Epoch 13/100
    255/255 - 1s - loss: 0.0045
    Epoch 14/100
    255/255 - 1s - loss: 0.0045
    Epoch 15/100
    255/255 - 1s - loss: 0.0044
    Epoch 16/100
    255/255 - 1s - loss: 0.0044
    Epoch 17/100
    255/255 - 1s - loss: 0.0044
    Epoch 18/100
    255/255 - 1s - loss: 0.0043
    Epoch 19/100
    255/255 - 1s - loss: 0.0043
    Epoch 20/100
    255/255 - 1s - loss: 0.0042
    Epoch 21/100
    255/255 - 1s - loss: 0.0042
    Epoch 22/100
    255/255 - 1s - loss: 0.0042
    Epoch 23/100
    255/255 - 1s - loss: 0.0042
    Epoch 24/100
    255/255 - 1s - loss: 0.0041
    Epoch 25/100
    255/255 - 1s - loss: 0.0041
    Epoch 26/100
    255/255 - 1s - loss: 0.0041
    Epoch 27/100
    255/255 - 1s - loss: 0.0040
    Epoch 28/100
    255/255 - 1s - loss: 0.0040
    Epoch 29/100
    255/255 - 1s - loss: 0.0040
    Epoch 30/100
    255/255 - 1s - loss: 0.0040
    Epoch 31/100
    255/255 - 1s - loss: 0.0040
    Epoch 32/100
    255/255 - 0s - loss: 0.0040
    Epoch 33/100
    255/255 - 0s - loss: 0.0040
    Epoch 34/100
    255/255 - 1s - loss: 0.0040
    Epoch 35/100
    255/255 - 1s - loss: 0.0040
    Epoch 36/100
    255/255 - 0s - loss: 0.0039
    Epoch 37/100
    255/255 - 1s - loss: 0.0040
    Epoch 38/100
    255/255 - 1s - loss: 0.0039
    Epoch 39/100
    255/255 - 1s - loss: 0.0039
    Epoch 40/100
    255/255 - 1s - loss: 0.0039
    Epoch 41/100
    255/255 - 1s - loss: 0.0040
    Epoch 42/100
    255/255 - 1s - loss: 0.0039
    Epoch 43/100
    255/255 - 1s - loss: 0.0039
    Epoch 44/100
    255/255 - 1s - loss: 0.0039
    Epoch 45/100
    255/255 - 1s - loss: 0.0039
    Epoch 46/100
    255/255 - 1s - loss: 0.0039
    Epoch 47/100
    255/255 - 1s - loss: 0.0039
    Epoch 48/100
    255/255 - 1s - loss: 0.0039
    Epoch 49/100
    255/255 - 1s - loss: 0.0039
    Epoch 50/100
    255/255 - 1s - loss: 0.0039
    Epoch 51/100
    255/255 - 1s - loss: 0.0039
    Epoch 52/100
    255/255 - 1s - loss: 0.0039
    Epoch 53/100
    255/255 - 1s - loss: 0.0039
    Epoch 54/100
    255/255 - 1s - loss: 0.0039
    Epoch 55/100
    255/255 - 1s - loss: 0.0040
    Epoch 56/100
    255/255 - 1s - loss: 0.0039
    Epoch 57/100
    255/255 - 1s - loss: 0.0039
    Epoch 58/100
    255/255 - 1s - loss: 0.0039
    Epoch 59/100
    255/255 - 1s - loss: 0.0039
    Epoch 60/100
    255/255 - 1s - loss: 0.0039
    Epoch 61/100
    255/255 - 1s - loss: 0.0039
    Epoch 62/100
    255/255 - 1s - loss: 0.0039
    Epoch 63/100
    255/255 - 1s - loss: 0.0039
    Epoch 64/100
    255/255 - 1s - loss: 0.0039
    Epoch 65/100
    255/255 - 1s - loss: 0.0039
    Epoch 66/100
    255/255 - 1s - loss: 0.0039
    Epoch 67/100
    255/255 - 1s - loss: 0.0039
    Epoch 68/100
    255/255 - 1s - loss: 0.0039
    Epoch 69/100
    255/255 - 1s - loss: 0.0039
    Epoch 70/100
    255/255 - 1s - loss: 0.0039
    Epoch 71/100
    255/255 - 1s - loss: 0.0039
    Epoch 72/100
    255/255 - 1s - loss: 0.0039
    Epoch 73/100
    255/255 - 1s - loss: 0.0039
    Epoch 74/100
    255/255 - 1s - loss: 0.0039
    Epoch 75/100
    255/255 - 1s - loss: 0.0039
    Epoch 76/100
    255/255 - 1s - loss: 0.0039
    Epoch 77/100
    255/255 - 1s - loss: 0.0039
    Epoch 78/100
    255/255 - 1s - loss: 0.0039
    Epoch 79/100
    255/255 - 1s - loss: 0.0039
    Epoch 80/100
    255/255 - 1s - loss: 0.0039
    Epoch 81/100
    255/255 - 1s - loss: 0.0039
    Epoch 82/100
    255/255 - 1s - loss: 0.0039
    Epoch 83/100
    255/255 - 1s - loss: 0.0039
    Epoch 84/100
    255/255 - 0s - loss: 0.0039
    Epoch 85/100
    255/255 - 1s - loss: 0.0039
    Epoch 86/100
    255/255 - 1s - loss: 0.0039
    Epoch 87/100
    255/255 - 1s - loss: 0.0039
    Epoch 88/100
    255/255 - 1s - loss: 0.0039
    Epoch 89/100
    255/255 - 1s - loss: 0.0039
    Epoch 90/100
    255/255 - 1s - loss: 0.0039
    Epoch 91/100
    255/255 - 1s - loss: 0.0039
    Epoch 92/100
    255/255 - 1s - loss: 0.0039
    Epoch 93/100
    255/255 - 1s - loss: 0.0039
    Epoch 94/100
    255/255 - 1s - loss: 0.0039
    Epoch 95/100
    255/255 - 1s - loss: 0.0039
    Epoch 96/100
    255/255 - 1s - loss: 0.0039
    Epoch 97/100
    255/255 - 1s - loss: 0.0039
    Epoch 98/100
    255/255 - 1s - loss: 0.0039
    Epoch 99/100
    255/255 - 1s - loss: 0.0039
    Epoch 100/100
    255/255 - 1s - loss: 0.0039
    Train Score: 0.06 RMSE
    Test Score: 0.02 RMSE
    [0.00892857 0.00884956 0.00877193 0.00869565 0.00862069 0.00854701
     0.00847458 0.00840336 0.00555556 0.00552486 0.00549451 0.00546448
     0.00543478 0.00540541 0.00537634 0.00534759 0.00531915 0.00529101
     0.00526316 0.0052356  0.00520833 0.00518135 0.00515464 0.00512821]
    Train on 255 samples
    Epoch 1/100
    255/255 - 1s - loss: 0.0060
    Epoch 2/100
    255/255 - 1s - loss: 0.0055
    Epoch 3/100
    255/255 - 1s - loss: 0.0052
    Epoch 4/100
    255/255 - 1s - loss: 0.0049
    Epoch 5/100
    255/255 - 0s - loss: 0.0048
    Epoch 6/100
    255/255 - 0s - loss: 0.0046
    Epoch 7/100
    255/255 - 1s - loss: 0.0045
    Epoch 8/100
    255/255 - 1s - loss: 0.0044
    Epoch 9/100
    255/255 - 1s - loss: 0.0043
    Epoch 10/100
    255/255 - 0s - loss: 0.0042
    Epoch 11/100
    255/255 - 1s - loss: 0.0042
    Epoch 12/100
    255/255 - 1s - loss: 0.0041
    Epoch 13/100
    255/255 - 1s - loss: 0.0041
    Epoch 14/100
    255/255 - 1s - loss: 0.0041
    Epoch 15/100
    255/255 - 1s - loss: 0.0040
    Epoch 16/100
    255/255 - 1s - loss: 0.0040
    Epoch 17/100
    255/255 - 1s - loss: 0.0040
    Epoch 18/100
    255/255 - 1s - loss: 0.0040
    Epoch 19/100
    255/255 - 1s - loss: 0.0039
    Epoch 20/100
    255/255 - 1s - loss: 0.0040
    Epoch 21/100
    255/255 - 1s - loss: 0.0039
    Epoch 22/100
    255/255 - 1s - loss: 0.0040
    Epoch 23/100
    255/255 - 1s - loss: 0.0039
    Epoch 24/100
    255/255 - 1s - loss: 0.0039
    Epoch 25/100
    255/255 - 1s - loss: 0.0039
    Epoch 26/100
    255/255 - 1s - loss: 0.0039
    Epoch 27/100
    255/255 - 1s - loss: 0.0039
    Epoch 28/100
    255/255 - 1s - loss: 0.0040
    Epoch 29/100
    255/255 - 1s - loss: 0.0039
    Epoch 30/100
    255/255 - 1s - loss: 0.0039
    Epoch 31/100
    255/255 - 1s - loss: 0.0039
    Epoch 32/100
    255/255 - 1s - loss: 0.0039
    Epoch 33/100
    255/255 - 1s - loss: 0.0039
    Epoch 34/100
    255/255 - 1s - loss: 0.0039
    Epoch 35/100
    255/255 - 1s - loss: 0.0039
    Epoch 36/100
    255/255 - 0s - loss: 0.0039
    Epoch 37/100
    255/255 - 1s - loss: 0.0039
    Epoch 38/100
    255/255 - 1s - loss: 0.0039
    Epoch 39/100
    255/255 - 1s - loss: 0.0039
    Epoch 40/100
    255/255 - 1s - loss: 0.0039
    Epoch 41/100
    255/255 - 1s - loss: 0.0039
    Epoch 42/100
    255/255 - 1s - loss: 0.0039
    Epoch 43/100
    255/255 - 0s - loss: 0.0039
    Epoch 44/100
    255/255 - 1s - loss: 0.0040
    Epoch 45/100
    255/255 - 1s - loss: 0.0039
    Epoch 46/100
    255/255 - 1s - loss: 0.0039
    Epoch 47/100
    255/255 - 1s - loss: 0.0039
    Epoch 48/100
    255/255 - 1s - loss: 0.0039
    Epoch 49/100
    255/255 - 1s - loss: 0.0039
    Epoch 50/100
    255/255 - 1s - loss: 0.0039
    Epoch 51/100
    255/255 - 1s - loss: 0.0039
    Epoch 52/100
    255/255 - 1s - loss: 0.0039
    Epoch 53/100
    255/255 - 1s - loss: 0.0039
    Epoch 54/100
    255/255 - 1s - loss: 0.0039
    Epoch 55/100
    255/255 - 1s - loss: 0.0039
    Epoch 56/100
    255/255 - 1s - loss: 0.0039
    Epoch 57/100
    255/255 - 1s - loss: 0.0039
    Epoch 58/100
    255/255 - 1s - loss: 0.0039
    Epoch 59/100
    255/255 - 1s - loss: 0.0039
    Epoch 60/100
    255/255 - 1s - loss: 0.0039
    Epoch 61/100
    255/255 - 1s - loss: 0.0039
    Epoch 62/100
    255/255 - 1s - loss: 0.0039
    Epoch 63/100
    255/255 - 1s - loss: 0.0039
    Epoch 64/100
    255/255 - 1s - loss: 0.0039
    Epoch 65/100
    255/255 - 1s - loss: 0.0039
    Epoch 66/100
    255/255 - 1s - loss: 0.0039
    Epoch 67/100
    255/255 - 1s - loss: 0.0039
    Epoch 68/100
    255/255 - 1s - loss: 0.0039
    Epoch 69/100
    255/255 - 1s - loss: 0.0039
    Epoch 70/100
    255/255 - 1s - loss: 0.0039
    Epoch 71/100
    255/255 - 1s - loss: 0.0039
    Epoch 72/100
    255/255 - 1s - loss: 0.0039
    Epoch 73/100
    255/255 - 1s - loss: 0.0039
    Epoch 74/100
    255/255 - 1s - loss: 0.0039
    Epoch 75/100
    255/255 - 1s - loss: 0.0039
    Epoch 76/100
    255/255 - 1s - loss: 0.0039
    Epoch 77/100
    255/255 - 1s - loss: 0.0039
    Epoch 78/100
    255/255 - 1s - loss: 0.0039
    Epoch 79/100
    255/255 - 1s - loss: 0.0039
    Epoch 80/100
    255/255 - 1s - loss: 0.0039
    Epoch 81/100
    255/255 - 1s - loss: 0.0039
    Epoch 82/100
    255/255 - 1s - loss: 0.0039
    Epoch 83/100
    255/255 - 1s - loss: 0.0039
    Epoch 84/100
    255/255 - 1s - loss: 0.0039
    Epoch 85/100
    255/255 - 1s - loss: 0.0038
    Epoch 86/100
    255/255 - 1s - loss: 0.0039
    Epoch 87/100
    255/255 - 1s - loss: 0.0039
    Epoch 88/100
    255/255 - 1s - loss: 0.0039
    Epoch 89/100
    255/255 - 1s - loss: 0.0039
    Epoch 90/100
    255/255 - 1s - loss: 0.0038
    Epoch 91/100
    255/255 - 1s - loss: 0.0039
    Epoch 92/100
    255/255 - 1s - loss: 0.0038
    Epoch 93/100
    255/255 - 1s - loss: 0.0039
    Epoch 94/100
    255/255 - 1s - loss: 0.0039
    Epoch 95/100
    255/255 - 1s - loss: 0.0039
    Epoch 96/100
    255/255 - 1s - loss: 0.0039
    Epoch 97/100
    255/255 - 1s - loss: 0.0039
    Epoch 98/100
    255/255 - 1s - loss: 0.0039
    Epoch 99/100
    255/255 - 1s - loss: 0.0038
    Epoch 100/100
    255/255 - 1s - loss: 0.0039
    Train Score: 0.06 RMSE
    Test Score: 0.01 RMSE
    [0.01388889 0.01369863 0.01351351 0.01333333 0.01315789 0.01298701
     0.01282051 0.01265823 0.0125     0.01234568 0.01219512 0.01204819
     0.01190476 0.01176471 0.01162791 0.01149425 0.01136364 0.01123596
     0.01111111 0.01098901 0.01086957 0.01075269 0.0106383  0.01052632]
    

#show on chart


```python
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=list(range(len(result_total))), 
                         y=result_total,
                         mode='lines+markers',
                         name='result_total',
                         line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(result_death))), 
                         y=result_death,
                         mode='lines+markers',
                         name='result_death',
                         line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(result_rec))), 
                         y=result_rec,
                         mode='lines+markers',
                         name='result_rec',
                         line=dict(color='green', width=2)))
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="0238880e-25df-4836-8450-cfc1c11556db" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("0238880e-25df-4836-8450-cfc1c11556db")) {
                    Plotly.newPlot(
                        '0238880e-25df-4836-8450-cfc1c11556db',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "result_total", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [0.007532993331551552, 0.007636391092091799, 0.007625453174114227, 0.007736251689493656, 0.00774294463917613, 0.007772232871502638, 0.00786976795643568, 0.007886502891778946, 0.007950988598167896, 0.007988234981894493, 0.008035680279135704, 0.008099623024463654, 0.008127730339765549, 0.008246234618127346, 0.008225410245358944, 0.008071712218225002, 0.007115395274013281, 0.007131211459636688, 0.007189641240984201, 0.007231552619487047, 0.007340554613620043, 0.007345196790993214, 0.007414855062961578, 0.007548863999545574, 0.007968150079250336, 0.00808754377067089, 0.008063239976763725, 0.008179916068911552, 0.008167557418346405, 0.00818610843271017, 0.008301516063511372, 0.008294135332107544, 0.008357789367437363, 0.008387674577534199, 0.0084169777110219, 0.008490093052387238, 0.008508395403623581, 0.008633370511233807, 0.008540337905287743, 0.008266117423772812, 0.007561998907476664, 0.007554309442639351, 0.007629885338246822, 0.0076880245469510555, 0.007809672504663467, 0.00781066482886672, 0.007893025875091553, 0.008070656098425388, 0.008385877124965191, 0.008501376956701279, 0.008485076017677784, 0.008583045564591885, 0.008566558361053467, 0.008583653718233109, 0.008696826174855232, 0.00867941789329052, 0.008735662326216698, 0.008761659264564514, 0.008777043782174587, 0.008856778964400291, 0.008874557912349701, 0.008974759839475155, 0.008825570344924927, 0.008495880290865898, 0.007978761568665504, 0.007960258983075619, 0.00805282685905695, 0.008130413480103016, 0.008243650197982788, 0.00825744029134512, 0.008346576243638992, 0.008539341390132904, 0.008782275952398777, 0.008884978480637074, 0.008882462047040462, 0.008954125456511974, 0.008940786123275757, 0.008960776962339878, 0.009059769101440907, 0.009041869081556797, 0.009087111800909042, 0.00910986214876175, 0.00911873858422041, 0.009200295433402061, 0.009219126775860786, 0.009279115125536919, 0.009090532548725605, 0.008747727610170841, 0.00836750864982605, 0.008351095020771027, 0.008458021096885204, 0.008551133796572685, 0.00865014549344778, 0.008680484257638454, 0.008772576227784157, 0.008962755091488361, 0.009154953993856907, 0.00924236886203289, 0.009251846931874752, 0.009298569522798061, 0.009291039779782295, 0.009314651601016521, 0.009394564665853977, 0.009380832314491272, 0.009414094500243664, 0.00943390280008316, 0.009442617185413837, 0.009521187283098698, 0.00953854899853468, 0.009553873911499977, 0.009341038763523102, 0.009011163376271725, 0.008731440640985966, 0.008726842701435089, 0.00884434673935175, 0.008947032503783703, 0.009032715111970901, 0.009077130816876888, 0.009169559925794601, 0.009347181767225266, 0.00950287189334631, 0.009575740434229374, 0.009592745453119278, 0.009619667194783688, 0.009618031792342663, 0.009643979370594025, 0.009704623371362686, 0.009696127846837044, 0.009718350134789944, 0.00973589438945055, 0.00974829401820898, 0.009819992817938328, 0.009831851348280907, 0.009804937057197094, 0.009580526500940323, 0.009278384037315845, 0.009073745459318161, 0.009086606092751026, 0.009210704825818539, 0.009317346848547459, 0.00939260981976986, 0.009446393698453903, 0.009537207894027233, 0.009697520174086094, 0.00982601661235094, 0.009886343032121658, 0.009906355291604996, 0.009919376112520695, 0.009922448545694351, 0.009948731400072575, 0.00999241229146719, 0.009988224133849144, 0.010001500137150288, 0.0100177563726902, 0.010035240091383457, 0.010097304359078407, 0.010099748149514198, 0.010036779567599297, 0.00981090497225523, 0.009543880820274353, 0.009396970272064209, 0.009429261088371277, 0.009556308388710022, 0.009662460535764694, 0.009730027057230473, 0.009788391180336475, 0.009876048192381859, 0.01001759059727192]}, {"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "result_death", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [0.029167143628001213, 0.030266940593719482, 0.03318609297275543, 0.03557130694389343, 0.03702821955084801, 0.037288956344127655, 0.037332385778427124, 0.03740386292338371, 0.0349474735558033, 0.034129608422517776, 0.030746879056096077, 0.027875112369656563, 0.02600405365228653, 0.023846307769417763, 0.0202697291970253, 0.018446624279022217, 0.017320509999990463, 0.014615336433053017, 0.011707330122590065, 0.00957457534968853, 0.007978551089763641, 0.007438160479068756, 0.007659142836928368, 0.009869856759905815, 0.017739957198500633, 0.020818326622247696, 0.0253668874502182, 0.030003931373357773, 0.034523289650678635, 0.03810961917042732, 0.04134220629930496, 0.04384699836373329, 0.044520024210214615, 0.045199327170848846, 0.04359512776136398, 0.04179064929485321, 0.039949432015419006, 0.03720027953386307, 0.03334212303161621, 0.03008531779050827, 0.026531683281064034, 0.021990280598402023, 0.0174143984913826, 0.013276701793074608, 0.009611206129193306, 0.006816226989030838, 0.004958713427186012, 0.004654485732316971, 0.00657091848552227, 0.008256001397967339, 0.01141132041811943, 0.015600821934640408, 0.020681366324424744, 0.026013512164354324, 0.03168362379074097, 0.03717953711748123, 0.04185176640748978, 0.045980244874954224, 0.04844065010547638, 0.05003838241100311, 0.05062086880207062, 0.04973398894071579, 0.04743340238928795, 0.0444774255156517, 0.040490295737981796, 0.03543775528669357, 0.029970893636345863, 0.024368558079004288, 0.01884251832962036, 0.013720124959945679, 0.009183477610349655, 0.005583386868238449, 0.003186659887433052, 0.001560920849442482, 0.0012771449983119965, 0.0023796726018190384, 0.00490812212228775, 0.008668649941682816, 0.013731472194194794, 0.019832318648695946, 0.026572704315185547, 0.03368861973285675, 0.040433309972286224, 0.04667605459690094, 0.05187227949500084, 0.05546717345714569, 0.05724884942173958, 0.05736337602138519, 0.05562577396631241, 0.05216973274946213, 0.04745931178331375, 0.041790690273046494, 0.03549817577004433, 0.02893633022904396, 0.022377025336027145, 0.016131579875946045, 0.01044965535402298, 0.005386712029576302, 0.0012586601078510284, -0.0017362255603075027, -0.0034334082156419754, -0.003726119175553322, -0.0024003293365240097, 0.0006256699562072754, 0.005346184596419334, 0.011702045798301697, 0.019396010786294937, 0.02813049964606762, 0.03727548569440842, 0.04605283588171005, 0.05373421311378479, 0.05974550545215607, 0.06352958083152771, 0.064883291721344, 0.0639316514134407, 0.06088769808411598, 0.05610549449920654, 0.04999926686286926, 0.04297181963920593, 0.0354364737868309, 0.02775244414806366, 0.020174654200673103, 0.012995639815926552, 0.006449287757277489, 0.0007323715835809708, -0.003956502303481102, -0.007357420399785042, -0.009234199300408363, -0.009362159296870232, -0.007527397945523262, -0.0035866927355527878, 0.0025388766080141068, 0.010722866281867027, 0.020611416548490524, 0.03160800039768219, 0.0428847037255764, 0.05340617150068283, 0.06220746040344238, 0.06856328994035721, 0.0720425620675087, 0.07258040457963943, 0.07039909064769745, 0.06589540839195251, 0.05956541746854782, 0.05192098766565323, 0.04343496263027191, 0.03456096351146698, 0.025676945224404335, 0.01707535609602928, 0.008998749777674675, 0.0016822479665279388, -0.004645815119147301, -0.009732929989695549, -0.013293756172060966, -0.015031831339001656, -0.014654716476798058, -0.011916691437363625, -0.006641944870352745, 0.0012228451669216156, 0.011519167572259903, 0.02375924214720726, 0.037104059010744095, 0.050408922135829926, 0.0623808354139328, 0.07185937464237213, 0.0780545026063919, 0.08066044747829437, 0.07981975376605988]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "result_rec", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [0.0150824636220932, 0.01472665648907423, 0.014525370672345161, 0.014473442919552326, 0.014487173408269882, 0.014512763358652592, 0.014470251277089119, 0.014436746947467327, 0.014408021233975887, 0.014453865587711334, 0.014420854859054089, 0.014415760524570942, 0.014427833259105682, 0.014408119954168797, 0.01437340211123228, 0.014391985721886158, 0.014410953968763351, 0.014402643777430058, 0.014424755237996578, 0.0144368726760149, 0.014469281770288944, 0.014514457434415817, 0.014609516598284245, 0.014681201428174973, 0.014978684484958649, 0.014776678755879402, 0.014691446907818317, 0.014702522195875645, 0.014728385955095291, 0.01473926194012165, 0.014706481248140335, 0.014694188721477985, 0.014686370268464088, 0.01471623033285141, 0.01469263806939125, 0.014685015194118023, 0.01468458492308855, 0.014665508642792702, 0.014644334092736244, 0.014652149751782417, 0.014655088074505329, 0.014638719148933887, 0.01464303582906723, 0.014641478657722473, 0.014655804261565208, 0.01467573270201683, 0.014715779572725296, 0.014732337556779385, 0.014797312207520008, 0.014714249409735203, 0.014693369157612324, 0.014713749289512634, 0.014726880006492138, 0.014726747758686543, 0.014710400253534317, 0.014711028896272182, 0.014712510630488396, 0.01472640410065651, 0.014713842421770096, 0.01470947451889515, 0.014708667062222958, 0.014700304716825485, 0.014693493023514748, 0.0146974828094244, 0.014697245322167873, 0.014687963761389256, 0.01469026505947113, 0.0146902771666646, 0.014698164537549019, 0.014706339687108994, 0.014719193801283836, 0.014719321392476559, 0.014731081202626228, 0.014702719636261463, 0.014701263047754765, 0.01471332460641861, 0.014716693200170994, 0.014714146964251995, 0.014707987196743488, 0.014711258932948112, 0.014713173732161522, 0.014717856422066689, 0.014711572788655758, 0.014709650538861752, 0.014709719456732273, 0.014706898480653763, 0.014705227687954903, 0.014706697314977646, 0.014705982990562916, 0.014701979234814644, 0.01470377016812563, 0.014704519882798195, 0.014708135277032852, 0.014710619114339352, 0.014713617973029613, 0.014711588621139526, 0.014713337644934654, 0.014705085195600986, 0.014706723392009735, 0.014711752533912659, 0.014711678959429264, 0.014710025861859322, 0.01470836903899908, 0.014710748568177223, 0.01471150852739811, 0.014712540432810783, 0.014709710143506527, 0.014709144830703735, 0.014709539711475372, 0.0147086838260293, 0.014708295464515686, 0.014708645641803741, 0.01470820140093565, 0.014706796035170555, 0.014707965776324272, 0.01470849011093378, 0.01470978558063507, 0.014710208401083946, 0.01471057441085577, 0.01470939815044403, 0.014709723182022572, 0.014707675203680992, 0.014708778820931911, 0.014710451476275921, 0.014709869399666786, 0.01470922864973545, 0.014709021896123886, 0.01471019722521305, 0.014710278250277042, 0.014710300602018833, 0.014709203504025936, 0.014709170907735825, 0.01470948662608862, 0.014709188602864742, 0.014709058217704296, 0.014709091745316982, 0.014708936214447021, 0.014708537608385086, 0.014709148555994034, 0.014709355309605598, 0.014709696173667908, 0.01470962818711996, 0.014709561131894588, 0.01470912341028452, 0.014709271490573883, 0.014708847738802433, 0.014709290117025375, 0.014709750190377235, 0.014709390699863434, 0.014709231443703175, 0.014709316194057465, 0.014709771610796452, 0.014709659852087498, 0.014709558337926865, 0.014709202572703362, 0.014709282666444778, 0.014709444716572762, 0.014709305949509144, 0.014709251001477242, 0.014709237962961197, 0.01470921654254198, 0.014709128998219967, 0.014709384180605412, 0.014709437265992165, 0.014709479175508022, 0.014709392562508583, 0.014709332026541233, 0.014709222130477428]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('0238880e-25df-4836-8450-cfc1c11556db');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
def FinalChartCalc(df,startVal):
  finalist=[]
  start=startVal
  for item in df:
    percent = start*item
    start = start+percent
    finalist.append(start)
  return finalist
```

# Growth compare next 7 days


```python
growth_total = FinalChartCalc(result_total,1)
growth_death =FinalChartCalc(result_death,1)
growth_rec =FinalChartCalc(result_rec,1)
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=list(range(len(growth_total))), 
                         y=growth_total,
                         mode='lines+markers',
                         name='result_total',
                         line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(growth_death))), 
                         y=growth_death,
                         mode='lines+markers',
                         name='growth_death',
                         line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(growth_rec))), 
                         y=growth_rec,
                         mode='lines+markers',
                         name='growth_rec',
                         line=dict(color='green', width=2)))
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="5ea83383-dd8c-4584-a1e1-158529f718f4" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("5ea83383-dd8c-4584-a1e1-158529f718f4")) {
                    Plotly.newPlot(
                        '5ea83383-dd8c-4584-a1e1-158529f718f4',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "result_total", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [1.0075329933315516, 1.0152269093068171, 1.022968474564837, 1.030882416154488, 1.0388644816322723, 1.0469387783054511, 1.0551779435553093, 1.0634996074584995, 1.071955480711558, 1.0805185129816117, 1.089201214287619, 1.098023333521137, 1.106947771082767, 1.1160759221131287, 1.1252561044374765, 1.1343388478842966, 1.1424101171616623, 1.1505568852807704, 1.1588289765132833, 1.1672091092339254, 1.1757770714457718, 1.1844133854178787, 1.193195639005384, 1.2022029106090866, 1.2117822438265313, 1.2215825857640001, 1.2314324993044508, 1.241505513793291, 1.2516455813623915, 1.2618916878107465, 1.2723673019275188, 1.2829204885218541, 1.2936428677400897, 1.3044935231342418, 1.3154734160426351, 1.326641907752779, 1.337929501662957, 1.3494803427687239, 1.3610053608925123, 1.3722555910200342, 1.3826325863001065, 1.3930774207024943, 1.403706441689755, 1.414498171270179, 1.4255449387462444, 1.4366793924612786, 1.4480191400801863, 1.4597056045837111, 1.471946516422373, 1.484460088618583, 1.4970558453157203, 1.5099051438488038, 1.5228398343832394, 1.5359113641899167, 1.5492689183442612, 1.562715670715657, 1.5763670271269163, 1.590178617894497, 1.6041356852452349, 1.6183431604383587, 1.632705240537724, 1.6473583779602026, 1.6618972552081919, 1.6760165353441592, 1.689389071664811, 1.7028370461984406, 1.7165496981006645, 1.730505956905369, 1.7447716426796223, 1.7591789903410813, 1.773862111910171, 1.789009726062794, 1.8047213031586027, 1.8207562131007142, 1.8369290110604943, 1.8533771038802365, 1.869947752171806, 1.8867039369102463, 1.9037970389414327, 1.9210109225253966, 1.9384673635491523, 1.9561265340109586, 1.9739639405122618, 1.9921249919398571, 2.0104906447939115, 2.029146218945769, 2.047592238695219, 2.065504017856025, 2.0827871405916856, 2.1001806939108065, 2.117944066527175, 2.136054889613706, 2.154532075190855, 2.1732344569521262, 2.192299321886586, 2.2119483637958917, 2.23219864930323, 2.252829452593422, 2.2736722858524354, 2.2948141856744937, 2.316135395560804, 2.3377093898312356, 2.3596711518639792, 2.3818068312569576, 2.4042293858477364, 2.4269106521829276, 2.449827040414694, 2.4731523024776814, 2.4967425868957034, 2.5205961507603774, 2.544141137111817, 2.5670668085506256, 2.5894810000109305, 2.6120789933763806, 2.6351811257043782, 2.6587581768894126, 2.6827739820528778, 2.7071258724400855, 2.7319490253540937, 2.7574850494728724, 2.783689076645831, 2.810344960693391, 2.837303884536779, 2.8645978036364896, 2.89214959638414, 2.9200414274283406, 2.948379329710309, 2.9769671926321517, 3.0058984021499335, 3.0351635115386832, 3.064751177842502, 3.0948470123976835, 3.1252750881692486, 3.1559182136951742, 3.186153571776281, 3.215715928217087, 3.2448945160192038, 3.2743795942987983, 3.3045389382295682, 3.335328473691584, 3.3666559126657374, 3.398458669864006, 3.4308704767177582, 3.464141412380405, 3.4981801234459877, 3.5327643321345246, 3.567761150770159, 3.603151115504288, 3.6389031970502406, 3.675105667548559, 3.7118288385934117, 3.7489034169797675, 3.786398075018854, 3.824329288464416, 3.862707351062666, 3.901710282836395, 3.9411165740454126, 3.9806726923493185, 4.019726693859609, 4.058090486355881, 4.096224242017514, 4.134848609872013, 4.174362498328579, 4.214697111230655, 4.25570622816096, 4.297362745470794, 4.33980370704521, 4.383278083854912]}, {"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "growth_death", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [1.0291671436280012, 1.0603168844251978, 1.0955046591323148, 1.1344731916207758, 1.1764807140346611, 1.2203504520200077, 1.2659090458796964, 1.313258934305252, 1.3591540161838074, 1.4055414105420532, 1.4487574223003243, 1.4891416982433199, 1.5278654188602963, 1.5642993678686892, 1.5960072924384654, 1.6254482393086573, 1.6536018317920698, 1.6777697788900237, 1.697411983561194, 1.713663982497265, 1.7273365381323074, 1.7401847445042946, 1.7535130680250968, 1.7708199908331275, 1.8022342616767564, 1.8397537631861485, 1.886422589833007, 1.9430226837595084, 2.0101022186687767, 2.0867064487158746, 2.1729754972047766, 2.268253950275147, 2.3692366710563117, 2.4763245744965587, 2.5842802607003414, 2.692279010754881, 2.7998340280615728, 2.903988636553885, 3.00081378295628, 3.0910942192466573, 3.1731061520636374, 3.2428836467160327, 3.2993565148011337, 3.343161087357186, 3.375292897690874, 3.3982996602359985, 3.4151508543908125, 3.431046625316285, 3.4535917530112643, 3.482104611352134, 3.521840022801684, 3.576783621879703, 3.650756394227, 3.745725390097318, 3.864403544180588, 4.008080279188376, 4.175825518775435, 4.367830998683812, 4.579411571810907, 4.808557919258554, 5.051971298816499, 5.303225983520652, 5.554776035558314, 5.801838172935986, 6.036756316382076, 6.250685409447425, 6.438024037008233, 6.5949093996680945, 6.719174100913526, 6.81136200920569, 6.873913999713217, 6.912293720872616, 6.9343208500030755, 6.945144775994569, 6.954014732907783, 6.97056301124033, 7.004775385760599, 7.0654973314998735, 7.162517011645521, 7.30456633134718, 7.498668412620728, 7.751288201276292, 8.064698439803022, 8.44112674448681, 8.878987230229557, 9.371479555052481, 9.907985976958592, 10.476341502169479, 11.059096106563056, 11.636046194893067, 12.188284939181539, 12.697641780034512, 13.148384899807237, 13.528850907246634, 13.831586346765425, 14.054711686729316, 14.201578579955797, 14.278078394131418, 14.296049641822881, 14.271228475023323, 14.222229521929874, 14.16923579978909, 14.135224967422731, 14.144068953009079, 14.219685756576348, 14.386085170537264, 14.665117833677563, 15.07765492570737, 15.639681836195802, 16.359933537040288, 17.239021692246965, 18.268975756750944, 19.42959612879867, 20.69025228245141, 22.01301427902924, 23.353336046372107, 24.663586513459958, 25.896747757341906, 27.009578131212415, 27.966702338652485, 28.74284668331146, 29.32272367589023, 29.703791231204043, 29.895359528336176, 29.917254040135667, 29.79888635561204, 29.579643421248385, 29.306499098661565, 29.032126985666313, 28.813590612640237, 28.710245116504687, 28.783136886241333, 29.091774614245175, 29.691397298954183, 30.629882996587266, 31.94343645404498, 33.64941309963087, 35.74265763262523, 38.193291831129834, 40.94483442843648, 43.916627076698745, 47.00831768721238, 50.10594997902989, 53.09053180718898, 55.84704465431301, 58.27275895188422, 60.286721647732804, 61.83470049724055, 62.89055002732225, 63.45648635039845, 63.563235895524414, 63.26793285317906, 62.65215049212631, 61.81926807982872, 60.8900112685522, 59.99768541714253, 59.28271151307043, 58.88895901143559, 58.96097109034777, 59.64015239656066, 61.057157219046886, 63.32262558353068, 66.51465088600719, 70.66389037554241, 75.74175334772602, 81.65373823181879, 88.2399652958728, 95.28325759811504]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "growth_rec", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [1.0150824636220932, 1.0300312343719389, 1.0449928198552845, 1.0601174637848019, 1.0754755693157876, 1.0910836917512798, 1.1068719469352548, 1.1228515771364096, 1.1390296465023944, 1.1554930279133584, 1.1721562251595459, 1.1890537485988308, 1.2062092178197292, 1.2235884249199, 1.2411755533699231, 1.2590385342123773, 1.2771824805738112, 1.2955772848802902, 1.3142656701065967, 1.333239556248383, 1.3525305750550358, 1.3721618225154182, 1.3922084434369892, 1.4126477360250935, 1.4338073407514045, 1.4549942512235097, 1.4763702220165407, 1.4980765879750688, 1.520140838153058, 1.542546592152472, 1.565232024684345, 1.5882318394479578, 1.6115572003142542, 1.635273247268644, 1.6592997252353203, 1.6836665669119968, 1.708390511595981, 1.7334449274090569, 1.758830074057394, 1.7846007156904216, 1.81075419635669, 1.8372613184849091, 1.864164401798843, 1.8914585251022669, 1.9191793710150344, 1.947344734471276, 1.9760014303358626, 2.0051125504202494, 2.0347828268400345, 2.064723128848805, 2.0950608679892406, 2.125887068347103, 2.157194752110004, 2.1889632150906904, 2.2211637401249376, 2.2538393440892674, 2.286998979398594, 2.3206782505470067, 2.3548243446371844, 2.389462573331099, 2.424608382779869, 2.4602508648257024, 2.496400543744115, 2.533091347821232, 2.5703208127836215, 2.608073591736932, 2.6463868840941545, 2.6852630409117233, 2.724731478913643, 2.764802305598706, 2.8054979665570494, 2.846792992792742, 2.888729331536639, 2.9312015090032673, 2.9742938734331, 3.018055624667803, 3.0624714233570898, 3.107533077954188, 3.1532386346781953, 3.199626744710322, 3.2467034088833158, 3.294487923500295, 3.342955022388217, 3.3921287225346806, 3.4420259844042893, 3.4926475111246957, 3.5440076880095535, 3.5961283363590635, 3.6490129385054413, 3.7026606509549183, 3.757103722177122, 3.8123501285616106, 3.868422689975908, 3.925329582741412, 3.983085382640101, 4.041682896232376, 4.1011495413384, 4.161457294743681, 4.222658696085138, 4.2847814058571165, 4.347817734311419, 4.41177424562579, 4.476664249347206, 4.542519331543501, 4.609346643425374, 4.67716184228561, 4.7459615372799, 4.815770572892699, 4.886608341376019, 4.958483918450956, 5.031414764979583, 5.105420061834606, 5.180511608340447, 5.256700335922144, 5.334015704559279, 5.4124710218013545, 5.492087309993453, 5.572877058880405, 5.654857281537616, 5.738036828775691, 5.822441762135232, 5.908076344465065, 5.994976932672982, 6.083165749942461, 6.172648323660641, 6.263443219227772, 6.355572342684521, 6.44906406532452, 6.543931592179306, 6.640194793019312, 6.737866769536206, 6.836975203402868, 6.937543598720222, 7.039589235954394, 7.143134963854772, 7.248203991387293, 7.354817361565909, 7.462995969331305, 7.5727702857169845, 7.684160854527619, 7.797192526047312, 7.911886329008859, 8.028266704634015, 8.146355470363156, 8.26618242463545, 8.387768443300576, 8.511146562767514, 8.636343402539513, 8.763378751865655, 8.892281318155678, 9.02308069575094, 9.155808152011222, 9.290486975598277, 9.42714593575359, 9.565811735005026, 9.70651796374921, 9.8492954531274, 9.99417175333456, 10.141178534206233, 10.290347542490746, 10.441710492791257, 10.59529895939179, 10.751149282293854, 10.909292638199073, 11.069762651080184, 11.23259213548872, 11.397816062728339, 11.565469070997333]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('5ea83383-dd8c-4584-a1e1-158529f718f4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


# Real values predict 7 days.


```python
real_total = FinalChartCalc(result_total,df['inc_total'].tail(1).values[0])
real_death = FinalChartCalc(result_death, df['inc_death'].tail(1).values[0])
real_rec = FinalChartCalc(result_rec, df['inc_rec'].tail(1).values[0])
fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=list(range(len(real_total))), 
                         y=real_total,
                         mode='lines+markers',
                         name='real_total',
                         line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(real_rec))), 
                         y=real_rec,
                         mode='lines+markers',
                         name='real_rec',
                         line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=list(range(len(real_death))), 
                         y=real_death,
                         mode='lines+markers',
                         name='real_death',
                         line=dict(color='red', width=2)))

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="4f00e801-d060-4ed2-a435-bbd7c55a8efb" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("4f00e801-d060-4ed2-a435-bbd7c55a8efb")) {
                    Plotly.newPlot(
                        '4f00e801-d060-4ed2-a435-bbd7c55a8efb',
                        [{"line": {"color": "orange", "width": 2}, "mode": "lines+markers", "name": "real_total", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [20722.938606843352, 20881.187070622615, 21040.41558484957, 21203.18953546551, 21367.364658212577, 21533.43679218652, 21702.899943045602, 21874.05992620642, 22047.98032727533, 22224.10477500579, 22402.69057546775, 22584.143923862743, 22767.701755630354, 22955.44956602283, 23144.267556070015, 23331.081423284213, 23497.09128978107, 23664.654016454886, 23834.794388925213, 24007.15695872338, 24183.382805496636, 24361.01451127493, 24541.647903062738, 24726.90946540769, 24923.937191024095, 25125.510623993956, 25328.103645693947, 25535.285407700416, 25743.846317461674, 25954.588234891442, 26170.050666045216, 26387.108607917507, 26607.646503678174, 26830.822783825093, 27056.65722116493, 27286.370758659163, 27518.53399020371, 27756.11169006712, 27993.158262837198, 28224.552996100065, 28437.98703502059, 28652.8163890089, 28871.434092674877, 29093.398386685043, 29320.608300132757, 29549.62174414358, 29782.857673169274, 30023.224875077773, 30274.995949775373, 30532.375102707018, 30791.44462645374, 31055.7289986822, 31321.769713594473, 31590.62493865821, 31865.363112504765, 32141.93591527964, 32422.717013946418, 32706.793812854015, 32993.86277412399, 33286.082123896165, 33581.48138737991, 33882.86711788545, 34181.90274512209, 34472.30809895867, 34747.35442600183, 35023.95236620952, 35305.99419053447, 35593.04652162963, 35886.46314663447, 36182.79347333536, 36484.795917768395, 36796.35204565954, 37119.50776336614, 37449.313791055494, 37781.955899492255, 38120.26027260871, 38461.085366669715, 38805.72657436995, 39157.297496947394, 39511.35265450236, 39870.39673347897, 40233.61055153741, 40600.49032845621, 40974.02683421899, 41351.771582121175, 41735.47943127658, 42114.87716548327, 42483.28663926272, 42838.76590768979, 43196.51651235748, 43561.87356033095, 43934.37696957472, 44314.41572252552, 44699.086310591345, 45091.21245256331, 45495.35394655391, 45911.86181886884, 46336.196180941515, 46764.891575412905, 47199.738170952995, 47638.27281589463, 48082.00673004886, 48533.71625153833, 48989.00290529311, 49450.19000811625, 49916.69829409847, 50388.04256724944, 50867.79655736097, 51353.00152727085, 51843.62162883946, 52327.89490811587, 52799.43011826928, 53260.44520822483, 53725.240735765416, 54200.40539348767, 54685.33818226146, 55179.29526286361, 55680.1649443477, 56190.727553483026, 56715.952497558064, 57254.91692845147, 57803.17515154169, 58357.6662971525, 58919.04762519535, 59485.73289842903, 60059.41207934615, 60642.266053481675, 61230.26121805814, 61825.31833541988, 62427.24310532769, 63035.80222586463, 63654.8133509956, 64280.65801346516, 64910.9258192824, 65532.80666429461, 66140.8452115691, 66740.99040548304, 67347.43949553774, 67967.7568815058, 68601.03604688855, 69245.37881170894, 69899.49792176293, 70566.1439651309, 71250.46056984023, 71950.56877903713, 72661.89678334295, 73381.71134904068, 74109.61214369224, 74844.9609569294, 75589.57337013881, 76344.89555218935, 77107.44548043993, 77878.63560698785, 78658.80480513617, 79448.16479665697, 80250.37709737904, 81060.88569496611, 81874.47593624085, 82677.7386393045, 83466.80512336781, 84251.14020981628, 85045.56620784762, 85858.28786562226, 86687.89018379216, 87531.36570081468, 88388.15694884335, 89261.08264650594, 90155.26362872789]}, {"line": {"color": "green", "width": 2}, "mode": "lines+markers", "name": "real_rec", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [633.4114573001862, 642.7394902480899, 652.0755195896975, 661.5132974017164, 671.0967552530515, 680.8362236527986, 690.688094887599, 700.6593841331196, 710.754499417494, 721.0276494179355, 731.4254844995565, 741.9695391256704, 752.674551919511, 763.5191771500176, 774.493545302832, 785.6400453485234, 796.9618678780581, 808.4402257653011, 820.1017781465163, 831.941483098991, 843.9790788343423, 856.2289772496208, 868.738068704681, 881.4921872796582, 894.6957806288763, 907.9164127634699, 921.2550185383213, 934.7997908964428, 948.5678830075079, 962.5490735031423, 976.7047834030311, 991.0566678155255, 1005.6116929960946, 1020.4105062956338, 1035.4030285468398, 1050.607937753086, 1066.035679235892, 1081.6696347032514, 1097.509966211814, 1113.590846590823, 1129.9106185265746, 1146.4510627345833, 1163.238586722478, 1180.2701196638143, 1197.5679275133814, 1215.143114310076, 1233.024892529578, 1251.1902314622353, 1269.7044839481812, 1288.3872324016538, 1307.3179816252857, 1326.553530648592, 1346.0895253166423, 1365.9130462165906, 1386.006173837961, 1406.3957507117027, 1427.0873631447223, 1448.103228341332, 1469.4103910536028, 1491.0246457586054, 1512.9556308546375, 1535.1965396512376, 1557.753939296327, 1580.649001040448, 1603.880187176979, 1627.4379212438446, 1651.3454156747514, 1675.6041375289144, 1700.2324428421123, 1725.2366386935914, 1750.630731131598, 1776.3988275026702, 1802.567102878862, 1829.069741618038, 1855.9593770222536, 1883.2667097927085, 1910.9821681748233, 1939.1006406434126, 1967.620908039193, 1996.56708869924, 2025.942927143188, 2055.7604642641827, 2086.003933970246, 2116.6883228616393, 2147.824214268275, 2179.4120469418085, 2211.4607973179595, 2243.984081888054, 2276.9840736273936, 2310.460246195867, 2344.432722638522, 2378.906480222443, 2413.895758544965, 2449.405659630639, 2485.445278767421, 2522.0101272490006, 2559.1173137951596, 2596.749351920055, 2634.9390263571236, 2673.7035972548383, 2713.0382662103225, 2752.94712927049, 2793.438491592654, 2834.532062883142, 2876.2323054974304, 2918.548989586218, 2961.479999262655, 3005.0408374850417, 3049.243605018633, 3094.093965113394, 3139.602813347257, 3185.782118584792, 3232.6392436044366, 3280.181009615415, 3328.4257996449874, 3377.3819176040424, 3427.062481435912, 3477.47528474137, 3528.63094367947, 3580.534981156028, 3633.2036595723816, 3686.6396389461975, 3740.865605987937, 3795.8954279640925, 3851.7325539642366, 3908.388568798126, 3965.877141835138, 4024.215976762497, 4083.413313519884, 4143.481550844047, 4204.428864190589, 4266.272526923386, 4329.027205601415, 4392.703683235538, 4457.316217445374, 4522.879290625667, 4589.406033617123, 4656.90948486273, 4725.408658287393, 4794.916373225229, 4865.448136253518, 4937.017069301523, 5009.638423691621, 5083.325813506604, 5158.097832972516, 5233.967508619556, 5310.955455166925, 5389.078283184652, 5468.348341164164, 5548.783542529138, 5630.402354148581, 5713.224286854997, 5797.263872773319, 5882.539063910234, 5969.066522643131, 6056.867209379501, 6145.960362751492, 6236.36317408076, 6328.095405344683, 6421.176866514219, 6515.627347501738, 6611.466550660471, 6708.717152151357, 6807.398606236214, 6907.531894274028, 7009.137492544953, 7112.237223142475, 7216.852700302328]}, {"line": {"color": "red", "width": 2}, "mode": "lines+markers", "name": "real_death", "type": "scatter", "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], "y": [419.9001946002245, 432.60928884548076, 446.9659009259845, 462.8650621812766, 480.0041313261418, 497.90298442416326, 516.4908907189163, 535.809645196543, 554.5348386029935, 573.4608955011578, 591.0930282985324, 607.5698128832746, 623.369090895001, 638.2341420904254, 651.1709753148941, 663.1828816379324, 674.6695473711646, 684.5300697871298, 692.5440892929673, 699.1749048588842, 704.7533075579815, 709.9953757577523, 715.4333317542396, 722.4945562599161, 735.3115787641167, 750.6195353799487, 769.660416651867, 792.7532549738795, 820.1217052168611, 851.3762310760771, 886.5740028595491, 925.4476117122601, 966.6485617909752, 1010.3404263945961, 1054.3863463657394, 1098.4498363879916, 1142.3322834491219, 1184.8273637139853, 1224.3320234461626, 1261.1664414526365, 1294.6273100419644, 1323.0965278601416, 1346.1374580388629, 1364.0097236417323, 1377.119502257877, 1386.506261376288, 1393.381548591452, 1399.8670231290446, 1409.0654352285962, 1420.698681431671, 1436.9107293030877, 1459.3277177269194, 1489.5086088446167, 1528.2559591597067, 1576.6766460256808, 1635.2967539088584, 1703.7368116603782, 1782.075047462996, 1868.3999212988513, 1961.8916310574914, 2061.204289917133, 2163.7162012764275, 2266.3486225077936, 2367.149974557884, 2462.996577083889, 2550.279647054552, 2626.7138070993615, 2690.7230350645846, 2741.423033172721, 2779.035699755924, 2804.556911882995, 2820.2158381160298, 2829.2029068012575, 2833.619068605787, 2837.2380110263784, 2843.9897085860575, 2857.9483573903276, 2882.7229112519517, 2922.3069407513763, 2980.263063189653, 3059.456712349261, 3162.5255861207315, 3290.396963439637, 3443.9797117506228, 3622.6267899336635, 3823.5636584614167, 4042.45827859911, 4274.347332885152, 4512.1112114777325, 4747.506847516378, 4972.820255186075, 5180.637846254088, 5364.541039121359, 5519.771170156633, 5643.2872294803, 5734.322368185568, 5794.244060621972, 5825.455984805625, 5832.788253863742, 5822.661217809523, 5802.669644947396, 5781.048206313956, 5767.171786708481, 5770.78013282771, 5801.631788683157, 5869.52274957921, 5983.368076140451, 6151.683209688613, 6380.990189167893, 6674.852883112444, 7033.520850436769, 7453.742108754393, 7927.275220549866, 8441.622931240185, 8981.30982584394, 9528.16110691983, 10062.743297491674, 10565.873084995508, 11019.907877534675, 11410.414554170224, 11727.081446791086, 11963.671259763225, 12119.14682233126, 12197.30668756117, 12206.239648375362, 12157.945633089723, 12068.494515869352, 11957.05163225393, 11845.107810151867, 11755.944969957229, 11713.780007533926, 11743.519849586477, 11869.444042612044, 12114.09009797332, 12496.992262607619, 13032.922073250367, 13728.960544649412, 14583.004314111113, 15582.863067100992, 16705.492446802105, 17917.98384729311, 19179.393616382677, 20443.22759144422, 21660.93697733313, 22785.594218959737, 23775.28565236879, 24596.982432275014, 25228.557802874177, 25659.34441114751, 25890.2464309626, 25933.800245373994, 25813.31660409709, 25562.077400787566, 25222.261376570146, 24843.124597569327, 24479.055650194183, 24187.346297332766, 24026.695276665752, 24056.07620486192, 24333.18217779678, 24911.320145371163, 25835.631238080554, 27137.97756149097, 28830.867273221338, 30902.635365872255, 33314.725198582106, 36001.90584071615, 38875.56910003099]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('4f00e801-d060-4ed2-a435-bbd7c55a8efb');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>
