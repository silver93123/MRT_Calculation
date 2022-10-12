import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib
matplotlib.use('Qt5Agg')
def output_open(fileName, list_name_result,list_name_new):
    output_csv = "./" + fileName + ".csv"
    df = pd.read_csv(output_csv)[list_name_result]
    df_result = df.rename(columns=dict(zip(list_name_result, list_name_new)))
    df_result.index = pd.date_range('2022-01-01 01:00', '2023-1-1 00:00', freq='1H')
    return df_result

list_name_result = [
    'WALL_N_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'WALL_E_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'WALL_S_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'WIN_S_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'WALL_W_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'FLOOR_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'CEILING_ZONE5:Surface Inside Face Temperature [C](Hourly)',
    'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)',
    'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)',
    'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
    'Environment:Site Solar Azimuth Angle [deg](Hourly)',
    'Environment:Site Solar Altitude Angle [deg](Hourly)',
    'ZONE_5:Zone Exterior Windows Total Transmitted Beam Solar Radiation Rate [W](Hourly)',
    'ZONE_5:Zone Exterior Windows Total Transmitted Diffuse Solar Radiation Rate [W](Hourly)',
    'ZONE_5:Zone Mean Radiant Temperature [C](Hourly)',
    'WALL_S_ZONE5:Surface Outside Face Beam Solar Incident Angle Cosine Value [](Hourly)',
    'WALL_S_ZONE5:Surface Outside Face Beam Solar Incident Angle Cosine Value [](Hourly)',
    'WALL_S_ZONE5:Surface Outside Face Beam Solar Incident Angle Cosine Value [](Hourly)',
    'ZONE_5:Zone Air Relative Humidity [%](Hourly)']
        # 'WALL_E_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'FLOOR_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'WIN_S_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'CEILING_ZONE5:Surface Inside Face Temperature [C](Hourly)',]
list_name_new = ['WALL_N','WALL_E','WALL_S','WIN_S', 'WALL_W','FLOOR','CEILING',
                 'Outdoor_AirTemperature', 'Site_Diffuse_Solar', 'Site_Beam_Solar',
                 'Solar_Azimuth','Solar_Altitude',
                 'Indoor_Beam_Solar', 'Indoor_Diffuse_Solar', 'MRT',
                 'cos_solar_S','cos_solar_W','cos_solar_E', 'Relative_Humidity']#, 'WALL_E', 'WIN_S', 'CEILING']
fileName = 'West_controlled_v3'
output = output_open(fileName, list_name_result,list_name_new)

#%% 읽기 및 재정렬
mrt_1 = pd.read_csv('./df_MC_MRT_total_1m_v3.csv', index_col=0)
mrt_2 = pd.read_csv('./df_MC_MRT_total_2m_v3.csv', index_col=0)
mrt_3 = pd.read_csv('./df_MC_MRT_total_3m_v3.csv', index_col=0)
list_name = ['Case A','Case B','Case C']

mrt_c_l = []
for df in [mrt_1, mrt_2, mrt_3]:
    mrt_c_r = pd.DataFrame()
    for c in mrt_1['cluster'].unique():
        mrt_c = df[df['cluster'] == c][df.columns[:-2]]
        mrt_c_r[c] = (mrt_c.stack()).reset_index()[0]
    mrt_c_l.append(mrt_c_r)
#%% PMV
from pythermalcomfort.utilities import v_relative, clo_dynamic
import pythermalcomfort

k_range= [2,3,4,5,6]
tdb, v = 25, 0.1
tr = 25
rh = pd.Series()
for i in range(len(k_range)):
    rh = pd.concat((rh,output['Relative_Humidity']))

met, clo = 1.4, 0.5
v_r = v_relative(v=v, met=met)
clo_d = clo_dynamic(clo=clo, met=met)
df_PMV_MC_l, df_PPD_MC_l = [], []
for m in [mrt_1,mrt_2,mrt_3]:
    mrt_df = m
    df_PMV_MC = pd.DataFrame()
    df_PPD_MC = pd.DataFrame()
    for i,c_name in enumerate(mrt_df.columns[:-2]):
        df_PMV_MC[i] = pythermalcomfort.models.pmv_ppd(tdb, mrt_df[c_name],
                                                       vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('pmv')
        df_PPD_MC[i] = pythermalcomfort.models.pmv_ppd(tdb,  mrt_df[c_name],
                                                       vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('ppd')
    df_PMV_MC['PMV'] = pythermalcomfort.models.pmv_ppd(tdb, mrt_df['MRT'],
                                                       vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('pmv')
    df_PPD_MC['PPD'] = pythermalcomfort.models.pmv_ppd(tdb,  mrt_df['MRT'],
                                                       vr=v, rh=rh, met=met, clo=clo, wme=0, standard='ISO').get('ppd')
    df_PMV_MC['cluster'] = np.array(mrt_df['cluster'])
    df_PPD_MC['cluster'] = np.array(mrt_df['cluster'])
    df_PMV_MC_l.append(df_PMV_MC)
    df_PPD_MC_l.append(df_PPD_MC)
#%%
plt.plot(df_PMV_MC_l[0][df_PMV_MC_l[0].columns[:-2]], alpha=0.3, color='r')
plt.plot(df_PMV_MC_l[0]['PMV'], alpha=1, color='k')


#%% PMV
df_PMV_error_l, df_PPD_error_l = [], []
for pmv, ppd in zip(df_PMV_MC_l, df_PPD_MC_l):
    df_PMV_error,df_PPD_error = pd.DataFrame(),pd.DataFrame()
    for i in pmv.columns[:-2]:
        df_PMV_error[i] =  np.array(pmv[i]) - np.array(pmv['PMV'])
        df_PPD_error[i] = np.array(ppd[i]) - np.array(ppd['PPD'])
    df_PMV_error['cluster'] = np.array(pmv['cluster'])
    df_PPD_error['cluster'] = np.array(ppd['cluster'])
    df_PMV_error_l.append(df_PMV_error)
    df_PPD_error_l.append(df_PPD_error)

df_PMV_error_l_r = []
for df in df_PMV_error_l:
    PMV_c_r = pd.DataFrame()
    for c in df['cluster'].unique():
        PMV_c = df[df['cluster'] == c][df.columns[:-1]]
        PMV_c_1 = pd.Series()
        for t in PMV_c.columns:
            PMV_c_2 = PMV_c[t]
            PMV_c_1 = pd.concat((PMV_c_1, PMV_c_2))
        PMV_c_r[c] = PMV_c_1.reset_index(drop=True)
    df_PMV_error_l_r.append(PMV_c_r)

#%%
k_range = [2,3,4,5,6]
c_iter = int(len(df_PMV_error_l_r[0]) / 8760)
time_range = pd.date_range('2021-1-1 01:00:00', '2022-1-1 00:00:00', freq='h')
index_new = []
for i in range(c_iter*len(k_range)):
    index_new += list(time_range)
df_empty_3_l,df_empty_4_l =[], []
df_empty_total = pd.DataFrame()
for i,df in enumerate(df_PMV_error_l_r):
    df_empty = df
    df_empty_3 = df_empty.copy().stack()
    df_empty_3.name = 'Error'
    df_empty_3 = df_empty_3.reset_index()
    df_empty_3['level_0'] = list_name[i]
    df_empty_3['level_1'] = df_empty_3['level_1'].astype('int')
    df_empty_3 = df_empty_3.sort_values('level_1')
    df_empty_3['Time'] = index_new
    df_empty_3_l.append(df_empty_3)
    df_empty_total = pd.concat((df_empty_total, df_empty_3))
    plt.plot(df.std(), label=list_name[i])
df_empty_total = df_empty_total.reset_index(drop=True)
#%%
dd = df_empty_3_l[0]
sns.boxplot(data=df_empty_total, y='Error', x='level_1', hue='level_0')

#%% std plot
sns.set_style("white")
for i, df in enumerate(df_PMV_error_l_r):
    plt.plot(df.std())
    plt.scatter(range(2,7),df.std(),ec='k', label=list_name[i])
plt.xlabel('Count of Input Surface Temperature')
plt.ylabel('Standard deviation')
plt.grid()
plt.legend()

#%% Dist 그래프
for i,df in enumerate(df_empty_3_l):
    sns.displot(df,bins=100, element='step', hue='level_1', x='Error')
    # plt.xlim(-0.2,0.2)
    plt.title(list_name[i])

