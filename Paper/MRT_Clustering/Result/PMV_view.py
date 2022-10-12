import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib
matplotlib.use('Qt5Agg')

#%% 읽기 및 재정렬
PMV_1 = pd.read_csv('./df_PMV_MC_1m_v2.csv', index_col=0)
PMV_2 = pd.read_csv('./df_PMV_MC_2m_v2.csv', index_col=0)
PMV_3 = pd.read_csv('./df_PMV_MC_3m_v2.csv', index_col=0)
list_name = ['Case A','Case B','Case C']
list_Dis = [1,2,3]

PMV_c_r_l = []
for df in [PMV_1, PMV_2, PMV_3]:
    PMV_c_r = pd.DataFrame()
    for c in PMV_1['cluster'].unique():
        PMV_c = df[df['cluster'] == c][df.columns[:-2]]
        PMV_c_r[c] = (PMV_c.stack()).reset_index()[0]
    PMV_c_r_l.append(PMV_c_r)
#%% 데이터 확인
for i,df in enumerate([PMV_1, PMV_2, PMV_3]):
    plt.plot(df['PMV'], label=list_name[i])
#%%


#%% 분산 분포
time_index = pd.date_range('2021-01-01 01:00:00', '2022-01-01 00:00:00', freq='h')
list_color = ['g','b','r']
k_range= [2,3,4,5,6]
c_iter = 70
df_std_total = pd.DataFrame()
for m,j in enumerate([PMV_1, PMV_2, PMV_3]):
    df_std = pd.DataFrame()
    for i in k_range:
        df = j[j['cluster']==i]
        # df_MC_MRT['Time'] = pd.date_range('2022-01-01 01:00', '2023-01-01 00:00', freq='h')
        df_2 = df[df.columns[:c_iter]]
        std_t = ((df_2.T).std()).reset_index(drop=True)
        df_std[i] = std_t
    df_std['Position'] = list_name[m]
    df_std['Time'] = time_index
    df_std['hour'] = df_std['Time'].dt.hour
    df_std['month'] = df_std['Time'].dt.month
    df_std_total = pd.concat((df_std_total, df_std))
df_std_total = df_std_total.reset_index(drop=True)
#%%
sns.boxplot(data=df_std_total, x='month', y=2, hue='Position')

#%%
sns.histplot(data=df_std_total, x=5, hue="Position", element="step",kde=True)
plt.xlabel('Standard deviation')
plt.grid()

#%% 표준오차
time_range = pd.date_range('2021-01-01 00:00:00', '2021-12-31 23:00:00', freq='h')

d_std_mean = pd.DataFrame()
for i in df_std_total['Position'].unique():
    df = df_std_total[df_std_total['Position']==i]
    d_std_mean[i] = df.std()

std_total_sns = pd.DataFrame()
for p in df_std_total['Position'].unique():
    df = df_std_total[df_std_total['Position']==p]
    df_1 = pd.DataFrame()
    for i,c in enumerate(df.columns[:5]):
        df_2 = pd.DataFrame()
        df_2['Standard_deviation'] =np.array(df[c])
        df_2['Number of input surface'] = c
        df_2['Time'] = time_range
        df_1 = pd.concat((df_1, df_2))
    df_1['Position'] = p
    std_total_sns = pd.concat((std_total_sns,df_1))
std_total_sns = std_total_sns.reset_index(drop=True)
std_total_sns['Hour'] = std_total_sns['Time'].dt.hour
std_total_sns['Month'] = std_total_sns['Time'].dt.month
#%%

sns.pointplot(data=std_total_sns, x="Number of input surface", y="Standard_deviation", hue="Position",
              errorbar='se')
plt.grid()

#%%
y_ = 'Standard_deviation'
#%%
# total_sns_p = total_sns[total_sns['Position']==
paper_rc = {'lines.linewidth': 0.8, 'markersize': 2}
sns.set_context("paper", rc = paper_rc)
sns.catplot(data=std_total_sns, x="Hour", y=y_, hue='Position', col= 'Number of input surface',
            capsize=.2,aspect=.75, palette="YlGnBu_d",
              errorbar='sd',kind='point')
plt.ylabel(y_)

#%%

sns.catplot(data=std_total_sns, x="Number of input surface", y=y_,   hue= 'Position',kind="box")
plt.ylabel(y_)