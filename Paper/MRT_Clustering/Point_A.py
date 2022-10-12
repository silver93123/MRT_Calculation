import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import pvlib as pv
import random
import datetime
import MRT_cal.MRT_cal as MRT_cal

#%% """경계조건-기하"""
N_split = 40
FLOOR = MRT_cal.cal_span(0, 3, 0, 6, 0, N_split, 'xy',-1, 'FLOOR')
CEILING = MRT_cal.cal_span(0, 3, 0, 6, 2.4, N_split, 'xy',1, 'CEILING')
WALL_E = MRT_cal.cal_span(0, 6, 0, 2.4, 0, N_split, 'yz',-1, 'INWALL_E')
WALL_W = MRT_cal.cal_span(0, 6, 0, 2.4, 3, N_split, 'yz',1, 'WALL_W')
WALL_N = MRT_cal.cal_span(0, 3, 0, 2.4, 0, N_split, 'xz',1, 'WALL_N')
WALL_S = MRT_cal.cal_span(0, 3, 0, 2.4, 6, N_split, 'xz',-1, 'WALL_S')
WIN_S = MRT_cal.cal_span(0.8, 2.3, 0.8, 1.8, 6, N_split, 'xz',-1, 'WIN_S')
# WIN_W = cal_span(0.5, 4.5, 0.7, 1.9, 3, N_split, 'yz',1, 'WIN_W')
span_list = [WALL_N, WALL_E, WALL_S, WIN_S, WALL_W, FLOOR, CEILING] #['WALL_N','WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']
name_list =[n[2] for n in span_list]
#%% 경계조건 체크
MRT_cal.Check_GeoBC(span_list,FLOOR,N_split)

#%% 데이터 불러오기()
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
    'ZONE_5:Zone Air Relative Humidity [%](Hourly)',
    'ZONE_5:Zone Air Temperature [C](Hourly)']
list_name_new = ['WALL_N','WALL_E','WALL_S','WIN_S', 'WALL_W','FLOOR','CEILING',
                 'Outdoor_AirTemperature', 'Site_Diffuse_Solar', 'Site_Beam_Solar',
                 'Solar_Azimuth','Solar_Altitude',
                 'Indoor_Beam_Solar', 'Indoor_Diffuse_Solar', 'MRT',
                 'cos_solar_S','cos_solar_W','cos_solar_E', 'Relative_Humidity', 'Indoor_AirTemperature']
fileName = 'C:/Users/silve/PycharmProjects/SilverGit/Paper/MRT_Clustering/MRT_cal/West_controlled_v3.csv'
output = MRT_cal.output_open(fileName, list_name_result,list_name_new)
temp_surface = output[['WALL_N', 'WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']]

#%% mrt_real
A_window = 1.5
x_w, z_w, H_w, L_w= 0.8, 0.8, 1,1.5 # 음영계산을 위한 창호 디테일 위치
obj_p_sol = [x_w + L_w / 2, 1, 0.5]
obj_p = [x_w + L_w / 2, 5, 0.5] #
vf = MRT_cal.cal_VewFactor_point(np.array(obj_p), span_list,N_split)
vf_T = vf.T
vf_T['WALL_S'] = vf_T['WALL_S'] - vf_T['WIN_S']
vf_r = vf_T.T
df_mrt_real = MRT_cal.cal_MRT(vf_r, temp_surface,output,1, obj_p_sol, x_w, z_w, L_w, H_w )[0]
