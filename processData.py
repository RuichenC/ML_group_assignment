import os
import numpy as np
import pandas as pd
import dataframe_image as dfi
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')
#visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import csv
pd.set_option('display.max_columns', None)

history_table = pd.read_csv('hisdata_1.csv')
hisDatahead=history_table.head()
hisDatainfo=history_table.info()
print(hisDatainfo)

corr_matrix=history_table.corr()
mask=np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)]=True
with sns.axes_style("white"):
    f,ax=plt.subplots(figsize=(15,15))
    ax=sns.heatmap(corr_matrix,mask=mask,vmax=.3,square=True,cmap="RdYlGn")
    #plt.show()


def calculate_ml_scores(df):
    X = df.copy()
    y = X["Share"]

    X.drop('Share', axis=1, inplace=True)

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return X, y, mi_scores

#drop columns for mutual information
to_drop_mi = ['Rank','Player','Age','year','Tm','team','First','Pts Won','Pts Max','WS','WS/48']
master_table_mi = history_table.copy()
master_table_mi.drop(to_drop_mi, axis=1, inplace=True)

X, y, mi_scores = calculate_ml_scores(df=master_table_mi)


def plot_mi_scores(scores, figsize):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(width, scores)

    for index, value in enumerate(scores):
        plt.text(value + 0.005, index, str(round(value, 2)))

    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plot_mi_scores(mi_scores, figsize=(14, 11))
#plt.show()

def add_win_lose_col(df):
    rank_lst = []
    for i in list(df['Rank']):
        if i == '1':
            rank_lst.append('won')
        else:
            rank_lst.append('lost')
    master_table_rank = df.copy()
    master_table_rank['Win/Lose'] = rank_lst
    return master_table_rank


def show_feature_vs_share(feature, df):
    fig = px.scatter(data_frame=df,
                     x=feature,
                     y='Share',
                     color='Win/Lose',
                     color_discrete_sequence=['blue', 'gray'],
                     hover_data={
                         'Win/Lose': False,
                         'Player': True,
                         'year': True,
                         'seed': True,
                         'W/L%': True,
                         'W': True

                     })
    fig.update_layout(height=500,
                      title=f"{feature} vs. MVP share")
    fig.show()

features = ['win_shares',
            'player_efficiency_rating',
            'value_over_replacement_player',
            'box_plus_minus',
            'offensive_box_plus_minus',
            'usage_percentage',
            'seed',
            'W',
            'W/L%',
            'PTS']

master_table_rank = add_win_lose_col(df=history_table)

for feature in features:
    show_feature_vs_share(feature=feature, df=master_table_rank)

to_drop = [
    'Rank',
    'Player',
    'Age',
    'year',
    'Tm',
    'team',
    'First',
    'Pts Won',
    'Pts Max',
    'WS/48',
    'WS',
    'MP',
    'G',
    'W',
    'FG%',
    '3P%',
    'STL',
    'BLK',
    'three_point_attempt_rate',
    'total_rebound_percentage',
    'offensive_rebound_percentage',
    'block_percentage',
    'defensive_rebound_percentage',
    'steal_percentage',
    'turnover_percentage',
    'assist_percentage',
    'AST',
    'TRB',
    #'free_throw_attempt_rate', ######### Experiment
    'FT%',
    'win_shares',
    #'value_over_replacement_player',
    'box_plus_minus',
    #'offensive_box_plus_minus',
    'defensive_box_plus_minus',
    'offensive_win_shares',
    'defensive_win_shares',
    'true_shooting_percentage'
]

#run another Mutual Information Score analysis
master_table_mi2 = history_table.copy()
cleanUp_Data=master_table_mi2.drop(to_drop, axis=1, inplace=False)
outputpath = 'cleanUp_data.csv'
# outputpath是保存文件路径
cleanUp_Data.to_csv(outputpath, sep=',', index=False, header=True)
# df1是你想要输出的的DataFrame
# index是否要索引，header是否要列名，True就是需要

X, y, mi_scores2 = calculate_ml_scores(df=master_table_mi2)
plot_mi_scores(mi_scores2, figsize=(14,4))
plt.show()

