import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib
matplotlib.use('Qt5Agg')

#%%
error3 = pd.read_csv('./3m_v2.csv', index_col=0)
error4 = pd.read_csv('./4m_v3.csv', index_col=0)
error5 = pd.read_csv('./5m_v3.csv', index_col=0)
list_name = ['Case A','Case B','Case C']


#%%
k_range = [2,3,4,5,6]
c_iter = int(len(error3) / 8760)
time_range = pd.date_range('2021-1-1 01:00:00', '2022-1-1 00:00:00', freq='h')
df_empty_3_l,df_empty_4_l =[], []
df_empty_total = pd.DataFrame()
for i,df in enumerate([error3, error4, error5]):
    df_empty = df
    df_empty_3 = df_empty.copy().stack()
    df_empty_3.name = 'Error'
    df_empty_3 = df_empty_3.reset_index()
    df_empty_3['level_0'] = list_name[i]
    df_empty_3_l.append(df_empty_3)
    df_empty_total = pd.concat((df_empty_total, df_empty_3))
df_empty_total = df_empty_total.reset_index()
df_empty_total['RSME'] = df_empty_total['Error'].pow(2)

#%% std plot
sns.set_style("white")
for i, df in enumerate([error3, error4, error5]):
    plt.plot(df.std())
    plt.scatter(range(0,5),df.std(),ec='k', label=list_name[i])
plt.xlabel('Count of Input Surface Temperature')
plt.ylabel('Error')
plt.grid()
plt.legend()
#%% mean plot
for i, df in enumerate([error3, error4, error5]):
    plt.plot((df.pow(2)).mean())
    plt.scatter(range(0,5),(df.pow(2)).mean(),ec='k')
plt.grid()
plt.legend()
#%% error bar
sns.set_style("white")
df_result_sum = pd.DataFrame()
for i in df_empty_total['level_0'].unique():
    for j in df_empty_total['level_1'].unique():
        mean_error = df_empty_total[(df_empty_total['level_0']==i)&(df_empty_total['level_1']==j)]['Error'].std()
        mean_error_dict = {
            'Case': [i],
            'Count': [j],
            'error_mean': [round(mean_error,4)]}
        df_result_sum = pd.concat([df_result_sum, pd.DataFrame(mean_error_dict)])
df_result_sum = df_result_sum.reset_index()
sns.boxplot(data=df_result_sum,  x='Count', y='error_mean', hue='Case',palette="YlGnBu_d")
plt.xlabel('Count of Input Surface')
plt.ylabel('Absolute Error')
#%%

sns.barplot(data=df, x="island", y="body_mass_g", hue="sex")
#%%
sns.set_style("whitegrid")
g = sns.catplot(
    data=df_empty_total, y="Error",hue='level_0', x="level_1",
    palette='viridis',
    kind="box", height=6,
)
#%%
sns.set_theme(style="whitegrid")
g1 = sns.catplot(
    data=df_empty_total, y="RSME", x="level_1", errorbar="sd", capsize=.2,
    palette='YlGnBu_d', col="level_0", aspect=.75,
    kind="point", height=6,
)
g1.despine(left=True)

#%%
df_sum = pd.DataFrame()
for i, df in enumerate([error3, error4, error5]):
    df_sum[str(i+3)] = df['3']
#%% Error hist
sns.set_style("ticks")
for i,df in enumerate(df_empty_3_l):
    sns.displot(df,bins=80, element='step', hue='level_1', x='Error')
    plt.xlim(-2,2)
    plt.title('<'+list_name[i]+'>')

#%%
count_ = df_empty_3[(df_empty_3['Error'] > 2)|(df_empty_3['Error'] < -2)]
sns.displot(count_,bins=200, element='step', hue='level_1', x='Error')

#%% Error 그래프

index_new = []
for i in range(c_iter*len(k_range)):
    index_new += time_range
df_empty_4['Time'] = index_new
df_empty_4['month'] = df_empty_4['Time'].dt.month
df_empty_4['hour'] = df_empty_4['Time'].dt.hour

sns.lineplot(x="hour", y=0, hue='level_1',
             data=df_empty_4)
#%% mrt 그래프 - error
plt.style.use('default')

data_set = df_empty.iloc[1:,:][df_empty.columns[:]]

fig, (axs, axs1) = plt.subplots(nrows=2, ncols=1, figsize=(9, 4))
axs.scatter(range(1,len(data_set.columns)+1),data_set.mean(), c='r', edgecolor='k')
axs.violinplot(data_set)
axs.yaxis.grid(True)
axs.set_xlabel('Input Data')
axs.set_ylabel('Error')
axs.set_xticks([y + 1 for y in range(len(data_set.columns))],
              labels=[str(i+2) for i in range(len(data_set.columns))])
axs1.plot(df_empty.std())
axs1.scatter(range(len(k_range)),df_empty.std(),c='r')
axs1.grid()
axs1.set_xlabel('Input Data')
axs1.set_ylabel('Std of Error')
#%% PMV
from pythermalcomfort.utilities import v_relative, clo_dynamic
import pythermalcomfort
tdb = 25
tr = 25
rh, v = 50, 0.1
met, clo = 1.4, 0.5
v_r = v_relative(v=v, met=met)
clo_d = clo_dynamic(clo=clo, met=met)

mrt_df = dataset_list[0]
df_PMV_MC = pd.DataFrame()
df_PPD_MC = pd.DataFrame()
for i,c_name in enumerate(mrt_df.columns[1:]):
    df_PMV_MC[i] = pythermalcomfort.models.pmv_ppd(tdb, mrt_df[c_name], vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('pmv')
    df_PPD_MC[i] = pythermalcomfort.models.pmv_ppd(tdb,  mrt_df[c_name], vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('ppd')
#%% PMV
df_PPD_MC.index = pd.date_range(temp_surface_dc.index[0], temp_surface_dc.index[-1], freq='h')
df_PPD_MC_2 = df_PPD_MC.stack()
df_PPD_MC_2 = df_PPD_MC_2.reset_index()
df_PPD_MC_2['hour'] = df_PPD_MC_2.level_0.dt.hour
df_PPD_MC_2['month'] = df_PPD_MC_2.level_0.dt.month
dd = df_PPD_MC_2.groupby('month').std()

sr_pmv = pd.DataFrame()
df_PMV_MC.index = pd.date_range(temp_surface_dc.index[0], temp_surface_dc.index[-1], freq='h')
df_PMV_MC_2 = df_PMV_MC.stack()
df_PMV_MC_2 = df_PMV_MC_2.reset_index()
df_PMV_MC_2['hour'] = df_PMV_MC_2.level_0.dt.hour
df_PMV_MC_2['month'] = df_PMV_MC_2.level_0.dt.month
dd2 = df_PMV_MC_2.groupby('month').std()

dd = df_PMV_MC.groupby('hour').std()

for i,cl_n in enumerate(df_PMV_MC.columns):
   PMV_MC_s = df_PMV_MC[cl_n]
   sr_pmv = pd.concat([sr_pmv, PMV_MC_s], axis=0)

sr_pmv['hour'] = sr_pmv.index.hour
sr_pmv.reset_index(drop=True, inplace=True)
import seaborn as sns
sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
sns.boxplot(x="month", y=0, hue='hour',
             data=df_PMV_MC_2)