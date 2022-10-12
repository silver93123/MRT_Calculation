import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib
matplotlib.use('Qt5Agg')

#%% 읽기 및 재정렬
mrt_1 = pd.read_csv('./df_MC_MRT_total_1m_v4.csv', index_col=0)
mrt_2 = pd.read_csv('./df_MC_MRT_total_2m_v4.csv', index_col=0)
mrt_3 = pd.read_csv('./df_MC_MRT_total_3m_v4.csv', index_col=0)
list_name = ['Case A','Case B','Case C']
list_Dis = [1,2,3]

mrt_c_l = []
for df in [mrt_1, mrt_2, mrt_3]:
    mrt_c_r = pd.DataFrame()
    for c in mrt_1['cluster'].unique():
        mrt_c = df[df['cluster'] == c][df.columns[:-2]]
        mrt_c_r[c] = (mrt_c.stack()).reset_index()[0]
    mrt_c_l.append(mrt_c_r)
#%% 데이터 확인
for i,df in enumerate([mrt_1, mrt_2, mrt_3]):
    plt.plot(df['MRT'], label=list_name[i])
#%%
df = mrt_3
plt.plot(df[df.columns[:-2]], color='r', alpha=0.3)
plt.plot(df['MRT'], color='k')

#%% 분산 분포
list_color = ['g','b','r']
k_range= [2,3,4,5,6]
c_iter = 70
df_std_total = pd.DataFrame()
for m,j in enumerate([mrt_1, mrt_2, mrt_3]):
    df_std = pd.DataFrame()
    for i in k_range:
        df = j[j['cluster']==i]
        # df_MC_MRT['Time'] = pd.date_range('2022-01-01 01:00', '2023-01-01 00:00', freq='h')
        df_2 = df[df.columns[:c_iter]]
        std_t = (df_2.T).std()
        df_std[i] = std_t
    df_std['Position'] = list_name[m]
    df_std_total = pd.concat((df_std_total, df_std))
df_std_total = df_std_total.reset_index(drop=True)
#%%
# for m in list_Dis:
#     df = df_std_total[df_std_total['Position']==m]
#     df[df.columns[:5]].hist(bins=50, alpha=0.5, color=list_color[m-1])
sns.histplot(data=df_std_total, x=5, hue="Position", element="step",kde=True)
plt.xlabel('Standard deviation')
plt.grid()

#%%
sns.histplot(data=df_std_total, x=6, hue="Position", element="step",kde=True)
plt.xlabel('Standard deviation')
plt.grid()

#%% 표준오차
d_std_mean = pd.DataFrame()
for i in df_std_total['Position'].unique():
    df = df_std_total[df_std_total['Position']==i]
    d_std_mean[i] = df.std()

std_total_sns = pd.DataFrame()
for p in df_std_total['Position'].unique():
    df = df_std_total[df_std_total['Position']==p]
    df_1 = pd.DataFrame()
    for i,c in enumerate(df.columns[:-1]):
        df_2 = pd.DataFrame()
        df_2['Standard error'] =np.array(df[c])
        df_2['Number of input surface'] = c
        df_1 = pd.concat((df_1, df_2))
    df_1['Position'] = p
    std_total_sns = pd.concat((std_total_sns,df_1))
std_total_sns = std_total_sns.reset_index(drop=True)
#%%

sns.pointplot(data=std_total_sns, x="Number of input surface", y="Standard error", hue="Position",
              errorbar='se')
plt.grid()
