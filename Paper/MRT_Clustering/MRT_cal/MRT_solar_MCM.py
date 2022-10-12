import matplotlib.pyplot as plt
from sklearn import preprocessing
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
from pythermalcomfort import utilities
import pythermalcomfort
import pvlib as pv
import random
import datetime
from matplotlib import cm

#%% """함수"""
def cal_normal_vector(Plane, direction):
    return direction*np.cross((Plane[1] - Plane[-1]), (Plane[1] - Plane[-2])) \
                   / np.linalg.norm(np.cross((Plane[1] - Plane[-1]), (Plane[1] - Plane[-2])))
def cal_span(c_1_min, c_1_max, c_2_min, c_2_max, c_3, N_split, plane,direction, name):
    if plane == 'xy':
        basis_1 = np.array([1, 0, 0])
        basis_2 = np.array([0, 1, 0])
        basis_3 = np.array([0, 0, 1])
    elif plane == 'xz':
        basis_1 = np.array([1, 0, 0])
        basis_2 = np.array([0, 0, 1])
        basis_3 = np.array([0, 1, 0])
    else:  ##yz
        basis_1 = np.array([0, 1, 0])
        basis_2 = np.array([0, 0, 1])
        basis_3 = np.array([1, 0, 0])
    c_1 = np.linspace(c_1_min, c_1_max, N_split)
    c_1 = np.delete((c_1 + (c_1[1] - c_1[0])/2),-1)
    c_2 = np.linspace(c_2_min, c_2_max, N_split)
    c_2 = np.delete((c_2 + (c_2[1] - c_2[0])/2),-1)
    vector_span = np.array([0, 0, 0])
    for i in c_1:
        for j in c_2:
            c_sum = basis_1 * i + basis_2 * j + basis_3 * c_3
            vector_span = np.vstack([vector_span, c_sum])
    vector_span = np.delete(vector_span, 0, axis=0)
    area = (c_1_max-c_1_min)*(c_2_max-c_2_min)
    norm_vector = cal_normal_vector(vector_span, direction)
    return vector_span, area, name, norm_vector
def matrix_rotation_xyz(angle_x, angle_y):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle_x), -1*np.sin(angle_x)],
                           [0, np.sin(angle_x), np.cos(angle_x)]])

    rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-1*np.sin(angle_y), 0, np.cos(angle_y)]])
    rotation = rotation_x.dot(rotation_y)
    return rotation
def cal_VewFactor_point(point, span_list):
    name_list = [n[2] for n in span_list]
    point_1 = point
    normal_set = np.concatenate((np.eye(3), -1*np.eye(3)), axis=0)
    df_point_VF = pd.DataFrame(index=name_list)
    for j in normal_set:
        viewF_dA1_A2 = np.zeros(len(span_list))
        for i in range(len(span_list)):
            to_plane = span_list[i]
            p_2 = to_plane[0]
            A_2 = to_plane[1]
            norm_vector1 = j
            norm_vector2 = to_plane[3]

            dot_norm_v = np.dot(norm_vector1, norm_vector2)
            delta_vector = p_2 - point_1
            cos = np.dot(delta_vector, norm_vector1)
            suface_count = (cos > 0).sum()
            if (dot_norm_v == 1) or (suface_count == 0):
                viewfactor_r = np.zeros(shape=(len(cos), 1))
            else:
                dA_2 = A_2 / ((N_split - 1) ** 2)
                delta_vector = p_2 - point_1
                distance = np.sqrt(np.sum(np.power(delta_vector, 2), axis=1))
                delta_vector_norm_dir = preprocessing.normalize(delta_vector, axis=1)
                cos_1 = np.dot(delta_vector_norm_dir, norm_vector1)
                cos_1[cos_1 < 0] = 0
                cos_2 = np.dot(delta_vector_norm_dir, norm_vector2)
                viewfactor_r = cos_1 * cos_2 * dA_2 / (np.pi * np.power(distance, 2))
            viewF_dA1_A2[i] = abs(viewfactor_r.sum())
        viewF_dA1_A2_s = pd.Series(viewF_dA1_A2, index=name_list, name=str(j))
        df_point_VF = pd.concat([df_point_VF, viewF_dA1_A2_s], axis=1)
    return df_point_VF
def cal_mrt(temp, VewFactor, name_list):
    test_temp = pd.Series(temp, index=name_list, name='temp')
    Temp_pr_list = np.zeros((len(VewFactor.columns)))
    for i in range(len(VewFactor.columns)):
        Temp_pr = (VewFactor[VewFactor.columns[i]]*((test_temp+273)**4)).sum()
        Temp_pr_list[i]= Temp_pr**0.25
    mrt = (0.18*(Temp_pr_list[2]+Temp_pr_list[5])+0.22*(Temp_pr_list[0]+Temp_pr_list[3])+0.30*(Temp_pr_list[1]+Temp_pr_list[4]))/(2*0.7)
    return mrt - 273
def cal_mrt_c(temp_pr):
    mrt = (0.18*(temp_pr[2]+temp_pr[5])+0.22*(temp_pr[0]+temp_pr[3])+0.30*(temp_pr[1]+temp_pr[4]))/(2*0.7)
    return mrt
def cal_f_projected(surface_azimuth,solar_azimuth, solar_altitude):
    SHARP = solar_azimuth - surface_azimuth
    person_loc_x, person_loc_y = person_loc
    win_x0, win_y0, win_z0 = win_loc
    zone_d, zone_h = zone_size
    t = np.linspace(0, 2, 100)
    hx_1 = np.sin(SHARP) * t + win_x0
    hy_1 = np.cos(SHARP) * t + win_y0
    hz_1 = -np.sin(solar_altitude) / ((np.cos(SHARP) ** 2 + np.sin(solar_altitude) ** 2) ** 0.5) * t + win_z0
    a, b, c = 0, -1, 0
    x0, y0, z0 = 0, person_loc_y, 0
    p_1 = np.argmin(abs(b * (hy_1 - y0)))
    hx, hy, hz = hx_1[p_1], hy_1[p_1], hz_1[p_1]
    hx2, hy2, hz2 = hx + win_d, hy, hz + win_h
    if hx < 0:
        hx = 0
    if hz < 0:
        hz = 0
    if hx > zone_d:
        hx = zone_d
    if hz > zone_h:
        hz = zone_h

    count_down_ir = int(round(hz * N_s, 0))
    count_middle_ir = int(round((hz2 - hz) * N_s, 0))
    if count_middle_ir > int(round((zone_h * N_s) - count_down_ir, 0)):
        count_middle_ir = int(round((zone_h * N_s) - count_down_ir, 0))
    count_top_ir = int(round(zone_h * N_s, 0)) - count_down_ir - count_middle_ir
    count_Protected_ir = np.hstack(
        [np.linspace(0, 0, count_down_ir), np.linspace(1, 1, count_middle_ir),
         np.linspace(0, 0, count_top_ir)])
    count_Protected_ir.shape = (int(round(zone_h * N_s, 0)), 1)
    count_NoProtected = np.zeros((len(count_Protected_ir), 1))
    ir_area = np.zeros((len(count_Protected_ir), 1))
    count_left, count_win_d = int(round(hx * N_s, 0)), int(round(hx2 * N_s, 0))
    if count_win_d > int(round((zone_d * N_s) - count_left, 0)):
        count_win_d = int(round((zone_d * N_s) - count_left, 0))
    count_right = int(zone_d * N_s) - count_left - count_win_d
    for i in range(count_left):
        ir_area = np.concatenate((ir_area, count_NoProtected), axis=1)
    for i in range(count_win_d):
        ir_area = np.concatenate((ir_area, count_Protected_ir), axis=1)
    for i in range(count_right):
        ir_area = np.concatenate((ir_area, count_NoProtected), axis=1)
    ir_area = np.delete(ir_area, (0), axis=1)

    count_person = int(round(Person_h * N_s, 0))
    count_zone_h = int(round(zone_h * N_s, 0))
    count_NoProtected_top = count_zone_h - count_person
    count_Protected = np.hstack(
        [np.linspace(1, 1, count_person),
         np.linspace(0, 0, count_NoProtected_top)])

    count_Protected.shape = (count_zone_h, 1)
    count_NoProtected = np.zeros((len(count_Protected), 1))
    Person_area = np.zeros((len(count_Protected), 1))
    count_person_loc_x, count_person_d = int(round((person_loc_x - Person_d / 2) * N_s, 0)), int(
        round(Person_d * N_s, 0))
    for i in range(count_person_loc_x):
        Person_area = np.concatenate((Person_area, count_NoProtected), axis=1)
    for i in range(count_person_d):
        Person_area = np.concatenate((Person_area, count_Protected), axis=1)
    for i in range(count_person_loc_x):
        Person_area = np.concatenate((Person_area, count_NoProtected), axis=1)
    Person_area = np.delete(Person_area, (0), axis=1)
    f_projected = (Person_area * ir_area).sum() / Person_area.sum()*0.27
    return f_projected
def cal_q_long_wave(vf_r, temp):
    q_long_wave = np.zeros((len(vf_r.columns)))
    for i in range(len(vf_r.columns)):
        q_long_wave[i] = (vf_r[vf_r.columns[i]] * ((temp + 273) ** 4)).sum()
    return q_long_wave
def cal_q_sol(ir_dir, ir_dff, sol_azimuth, sol_altitude, A_window):
    a_dir, a_dff = 0.7, 0.7
    st =  5.6703*10**-8
    f_eff = 0.7
    q_sol_dff = (vf_r.T['WIN_S'] * (ir_dff * a_dff)) / st /A_window
    cos_set = np.linspace(0, 0, 6)
    for i, j, r in zip([90, 90, 0, 90, 90, 180], [90, 180, 0, 270, 0, 0], range(6)):
        In_angle = pv.irradiance.aoi(i, j, 90 - sol_altitude, sol_azimuth)
        cos_theta = np.cos(np.radians(In_angle))
        if cos_theta < 0:
            cos_theta = 0
        cos_set[r] = cos_theta
    q_sol_dir = abs(a_dir*ir_dir*cos_set/st)/A_window
    return q_sol_dir, q_sol_dff.array
def group_split(NumGroup, clust_df):
    group_i = clust_df.loc[str(NumGroup),:].reset_index()
    group_key = np.unique(group_i[str(NumGroup)])
    group_split = [group_i[(k == group_i[str(NumGroup)])]['index'].values for k in group_key]
    return group_split
def random_dataset(c_sample, group_i):
    df = pd.DataFrame(index=range(c_sample))
    for j in range(len(group_i)):
        randomSelect_index = np.zeros(c_sample)
        randomSelect_data = pd.Series(index=range(c_sample), data=np.nan, name=j)
        for i in range(c_sample):
            randomSelect_index[i] = random.randint(0, len(group_i[j]) - 1)
            randomSelect_data[i] = group_i[j][int(randomSelect_index[i])]
        df = pd.concat([df, randomSelect_data], axis=1)
    return df
def output_open(fileName, list_name_result,list_name_new):
    output_csv = "./" + fileName + ".csv"
    df = pd.read_csv(output_csv)[list_name_result]
    df_result = df.rename(columns=dict(zip(list_name_result, list_name_new)))
    df_result.index = pd.date_range('2022-01-01 01:00', '2023-1-1 00:00', freq='1H')
    return df_result

#%% """경계조건-기하"""
N_split = 40
FLOOR = cal_span(0, 3, 0, 6, 0, N_split, 'xy',-1, 'FLOOR')
CEILING = cal_span(0, 3, 0, 6, 2.4, N_split, 'xy',1, 'CEILING')
WALL_E = cal_span(0, 6, 0, 2.4, 0, N_split, 'yz',-1, 'INWALL_E')
WALL_W = cal_span(0, 6, 0, 2.4, 3, N_split, 'yz',1, 'WALL_W')
WALL_N = cal_span(0, 3, 0, 2.4, 0, N_split, 'xz',1, 'WALL_N')
WALL_S = cal_span(0, 3, 0, 2.4, 6, N_split, 'xz',-1, 'WALL_S')
WIN_S = cal_span(0.8, 2.3, 0.8, 1.8, 6, N_split, 'xz',-1, 'WIN_S')
# WIN_W = cal_span(0.5, 4.5, 0.7, 1.9, 3, N_split, 'yz',1, 'WIN_W')
span_list = [WALL_N, WALL_E, WALL_S, WIN_S, WALL_W, FLOOR, CEILING] #['WALL_N','WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']
name_list =[n[2] for n in span_list]

#%% 경계조건 그래프

c_l = ['grey', 'grey', 'grey', 'grey','grey','grey','grey','grey','grey', 'skyblue'
    ,'r','r','r','r','r','r' ]
ax = plt.figure().add_subplot(projection='3d')
mid_value= round(len(FLOOR[0])/2)
for i,c_g in zip(span_list, c_l):
    X = np.reshape(i[0][:,0], (N_split-1, N_split-1))
    Y = np.reshape(i[0][:,1], (N_split-1, N_split-1))
    Z = np.reshape(i[0][:,2], (N_split-1, N_split-1))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,color=c_g,alpha=0.1,
                           linewidth=0, antialiased=False)
    ax.quiver(i[0][mid_value,0], i[0][mid_value,1], i[0][mid_value,2], i[3][0], i[3][1],
              i[3][2],color =c_g, length=0.5, arrow_length_ratio=0.8)
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_zlim(0,6)

win_loc, person_loc = (0.2, 0, 1), (1.2, 0.7)
zone_size = (2.4, 2.3)
win_d, win_h, Person_d, Person_h = 2, 1.2, 0.4, 1.4
N_s = 100
surf_azimuth = 0
sol_azimuth = np.pi/8
sol_altitude = np.pi/6
ir_dir, ir_dff = 100, 50


#%% 데이터 불러오기
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
        # 'WALL_E_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'FLOOR_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'WIN_S_ZONE5:Surface Inside Face Temperature [C](Hourly)',
        # 'CEILING_ZONE5:Surface Inside Face Temperature [C](Hourly)',]
list_name_new = ['WALL_N','WALL_E','WALL_S','WIN_S', 'WALL_W','FLOOR','CEILING',
                 'Outdoor_AirTemperature', 'Site_Diffuse_Solar', 'Site_Beam_Solar',
                 'Solar_Azimuth','Solar_Altitude',
                 'Indoor_Beam_Solar', 'Indoor_Diffuse_Solar', 'MRT',
                 'cos_solar_S','cos_solar_W','cos_solar_E', 'Relative_Humidity', 'Indoor_AirTemperature']#, 'WALL_E', 'WIN_S', 'CEILING']
dataTime = ['2022-05-04 00:00', '2022-05-14 00:00']

def cal_mrt_AddSolar(Temp_surface, solar_beam, solar_diffuse, Solar_Azimuth, Solar_Altitude, vf, obj_p, x_w, z_w, H_w, L_w):
    T_sur = Temp_surface.values
    sol_rad = cal_q_sol(solar_beam, solar_diffuse, Solar_Azimuth, Solar_Altitude, 1)
    long_rad = cal_q_long_wave(vf, T_sur)
    C_shade = cal_Shading(obj_p, x_w, z_w, H_w, L_w, np.radians(Solar_Altitude), np.radians(Solar_Azimuth)-np.pi)[0]
    temp_pr_sol = (long_rad+ sol_rad[0]*C_shade + sol_rad[1]) ** 0.25 - 273
    mrt_sol = cal_mrt_c(temp_pr_sol)
    return mrt_sol, temp_pr_sol, C_shade
def cal_Shading(obj_p, x_w, z_w, H_w, L_w, angle_attitude, angle_azimuth):
    if (angle_attitude < 0.02):
        c_shading = 0
        line_v_shading = [np.linspace(0,0,3) for i in range(4)]
        A, B, C, D = [np.linspace(0,0,3) for i in range(4)]
    elif (abs(angle_azimuth)>(np.pi/2-0.02)):
        c_shading = 0
        line_v_shading = [np.linspace(0,0,3) for i in range(4)]
        A, B, C, D = [np.linspace(0,0,3) for i in range(4)]
    else:
        s, z_b = 0, 0
        A_x = x_w + ((z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth)+s/2)*np.tan(angle_azimuth) \
              #+ abs(s/2*np.tan(angle_azimuth))
        A_y = (z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth)
        B_x = x_w + ((H_w+z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth)-s/2)*np.tan(angle_azimuth) \
              #+ abs(s/2*np.tan(angle_azimuth))
        B_y = (H_w+z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth) #- s
        C_x = x_w + L_w + ((H_w+z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth)-s/2)*np.tan(angle_azimuth) \
              #- abs(s/2*np.tan(angle_azimuth))
        C_y = B_y
        D_x = x_w + L_w + ((z_w-z_b)/np.tan(angle_attitude)*np.cos(angle_azimuth)+s/2)*np.tan(angle_azimuth) \
              #- abs(s/2*np.tan(angle_azimuth))
        D_y = A_y
        A,B,C,D = np.array([A_x, A_y, 0]), np.array([B_x, B_y, 0]), np.array([C_x, C_y, 0]), np.array([D_x, D_y, 0])

        win_A0 = np.array([x_w, 0, z_w])
        win_B0 = win_A0 + np.array([0, 0, H_w])
        win_C0 = win_A0 + np.array([L_w, 0, H_w])
        win_D0 = win_A0 + np.array([L_w, 0, 0])
        win_v = [win_A0, win_B0, win_C0, win_D0]
        a_v = [s - w / np.linalg.norm(s - w) for s, w in zip([A,B,C,D], win_v)]
        t_v = [(obj_p[1] - win_v[i][1]) / a_v[i][1] for i in range(len(a_v))]
        line_v_shading = [win_v[i] + a_v[i] * t for i, t in enumerate(t_v)]

        if (line_v_shading[0][0] < obj_p[0] < line_v_shading[2][0]) & (
                line_v_shading[0][2] < obj_p[2] < line_v_shading[1][2]):
            c_shading = 1
        else:
            c_shading = 0
    return c_shading, line_v_shading, [A,B,C,D]

#%% Run
iddfile = 'C:/EnergyPlusV8-9-0/Energy+.idd'
IDF.setiddname(iddfile)
weatherData = "./KOR_DAEJEON_471330_IW2.epw"
fileName = 'West_controlled_v3' #West_controlled_v5_timeS West_controlled_v3
idf = IDF("./" + fileName + ".idf", weatherData)
idf.run()
#%% 일교차 분포
output = output_open(fileName, list_name_result,list_name_new)
output_maxDay = output.resample('D').max()
output_minDay = output.resample('D').min()

s = output[['WALL_N', 'WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']]
temp_surface =  s
temp_surface_dc = temp_surface[:]
#%%
output_meanDay = output.resample('D').mean()
output_meanDay_std = output_meanDay[['WALL_N', 'WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']].T.std()
output_meanDay_std.hist(color='grey', bins=30, alpha=0.5)
output_meanDay_std_95=output_meanDay_std.quantile(0.95)
output_meanDay_std_95_day = output_meanDay_std[(output_meanDay_std<output_meanDay_std_95+0.02)
                                               &(output_meanDay_std>output_meanDay_std_95-0.02)]
dataTime_std_95_day = ['2022-01-11 00:00', '2022-01-12 00:00']
output_std_95_day = output[dataTime_std_95_day[0]:dataTime_std_95_day[-1]]

output_meanDay_std_5=output_meanDay_std.quantile(0.05)
output_meanDay_std_5_day = output_meanDay_std[(output_meanDay_std<output_meanDay_std_5+0.02)
                                               &(output_meanDay_std>output_meanDay_std_5-0.02)]
dataTime_std_5_day = ['2022-01-01 00:00', '2023-01-01 00:00']
output_std_5_day = output[dataTime_std_5_day[0]:dataTime_std_5_day[-1]]

output_day= output_std_5_day
temp_surface =  output_day.iloc[:,:7]
temp_surface_dc = temp_surface[:]

#%% mrt_real
A_window = 1.5
x_w, z_w, H_w, L_w = 0.8, 0.8, 1, 1.5
dmrt_array = np.linspace(0,0,len(output_day))
df_mrt_real = pd.DataFrame()
df_c_shade = pd.DataFrame()
for i in [1.0]:
    obj_p = [x_w+L_w/2,i,0.5]
    obj_p_c = [x_w+L_w/2,6-i,0.5]
    vf = cal_VewFactor_point(np.array(obj_p), span_list)
    vf_T = vf.T
    vf_T['WALL_S'] = vf_T['WALL_S'] - vf_T['WIN_S']
    vf_r = vf_T.T
    mrt_real = np.linspace(0,0,len(temp_surface_dc))
    c_shade = np.linspace(0,0,len(temp_surface_dc))
    for d in range(len(temp_surface_dc)):
        mrt_sol_c = cal_mrt_AddSolar(temp_surface_dc.iloc[d], output_day['Indoor_Beam_Solar'][d],
                                       output_day['Indoor_Diffuse_Solar'][d]
                                       , output_day['Solar_Azimuth'][d],
                                     output_day['Solar_Altitude'][d],
                                     vf_r, obj_p_c,x_w, z_w, H_w, L_w)
        mrt_real[d] = mrt_sol_c[0]
        c_shade[d] = mrt_sol_c[2]
        print(str(i)+'m:'+str(round(d/len(temp_surface_dc),2)))
    df_mrt_real[str(i)+'m'] = mrt_real
    df_c_shade[str(i)+'m'] = c_shade
df_mrt_real.index = temp_surface_dc.index
df_c_shade.index = temp_surface_dc.index
#%% mrt_real 그래프
plt.plot(df_c_shade['4.0m']*20)
plt.plot(df_mrt_real)
plt.plot(output_day['Indoor_Beam_Solar']/20)
# plt.plot(output_day['Solar_Azimuth']/20)
# plt.plot(output_day['Solar_Altitude']/20)
output_day['Indoor Solar Irradiation[W/m²]'] = output_day['Indoor_Beam_Solar'] + output_day['Indoor_Diffuse_Solar']
output_day['hour'] = output_day.index.hour
output_day['month'] = output_day.index.month

sns.lineplot(x="hour", y='Indoor Solar Irradiation[W/m²]', color='r',
            ci = 'sd', palette='Paired', legend='brief',
            data=output_day, linewidth=1)
sns.lineplot(x="hour", y='Indoor Solar Irradiation[W/m²]', ci=1,
            hue = 'month', palette='Paired', legend='brief',
            data=output_day, linewidth=0.5)
plt.xlim(0,24)
#%% mrt_real 그래프
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")
day_serise = np.linspace(0,0,8760)
count = 0
for i,time in enumerate(day_serise):
    if i%24==0:
        count +=1
    day_serise[i] = count

df_mrt_real_s = df_mrt_real.stack()
df_mrt_real_s = df_mrt_real_s.reset_index()
df_mrt_real_s['month'] = df_mrt_real_s.level_0.dt.month
df_mrt_real_s['hour'] = df_mrt_real_s.level_0.dt.hour

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
df_mrt_real_s_win = df_mrt_real_s[df_mrt_real_s['level_1']=='5m']
df_mrt_real_s_win['day'] = day_serise

sns.boxplot(x="month", y=0,
            hue="hour",
            data=df_mrt_real_s_win, linewidth=0.5)
plt.legend([],[], frameon=False)
sns.despine(offset=10, trim=True)

out_temp = pd.DataFrame(index=output_day.index)
out_temp['Outdoor_AirTemperature'] = output_day[['Outdoor_AirTemperature']]
out_temp['month'] = out_temp.index.month
out_temp['hour'] = out_temp.index.hour
out_temp['day'] = day_serise

sns.boxplot(x="month", y='Outdoor_AirTemperature',
            hue="hour",
            data=out_temp, linewidth=0.5)
plt.legend([],[], frameon=False)

#%%

temp_surface_win = temp_surface
temp_surface_win['month'] = temp_surface_win.index.month
temp_surface_win['hour'] = temp_surface_win.index.hour
temp_surface_win['day'] = day_serise

sns.lineplot(x="month", y='WALL_S',
            palette='tab10', linewidth=2,
            data=temp_surface_win)
sns.despine(offset=10, trim=True)
#%%
output_day_SurfaceTemp = output_day[['WALL_N', 'WALL_E', 'WALL_S', 'WIN_S', 'WALL_W', 'FLOOR', 'CEILING']]
output_day_BC = output_day[['Outdoor_AirTemperature', 'Site_Diffuse_Solar', 'Site_Beam_Solar']]
output_day_ir = output_day[['Indoor_Beam_Solar', 'Indoor_Diffuse_Solar']]/1.5
#%% BC 그래프
output_day_solar_sum = (output_day_ir.resample('D').sum()/1000)[['Site_Diffuse_Solar', 'Site_Beam_Solar']]
output_day_solar_sum_s = output_day_solar_sum.stack()
output_day_solar_sum_s = output_day_solar_sum_s.reset_index()
output_day_BC_mean = output_day_BC.resample('h').mean()
output_day_solar_sum_s['month'] = output_day_solar_sum_s.level_0.dt.month
output_day_BC_mean['month'] = output_day_BC_mean.index.month

ax = sns.boxplot(x="month", y="Outdoor_AirTemperature", data=output_day_BC_mean, color='salmon')
ax.set(ylabel='Outdoor AirTemperature[C˚]',xlabel='Month')
plt.grid()

ax = sns.boxplot(x="month", y=0,hue="level_1", data=output_day_solar_sum_s)
ax.set(ylabel='Irradiation[kWh/m2]',xlabel='Month')
plt.grid()
#%% 출력데이터 그래프(표면온도)
output_day_SurfaceTemp_s = output_day_SurfaceTemp.stack()
output_day_SurfaceTemp_s = output_day_SurfaceTemp_s.reset_index()
output_day_SurfaceTemp_s['month'] = output_day_SurfaceTemp_s.level_0.dt.month
output_day_SurfaceTemp_s['hour'] = output_day_SurfaceTemp_s.level_0.dt.hour
output_day_SurfaceTemp_s['Surface'] = output_day_SurfaceTemp_s['level_1']
output_day_SurfaceTemp_s['day'] = output_day_SurfaceTemp_s.level_0.dt.day
#%%
ax = sns.lineplot(x="hour", y=0,data=output_day_SurfaceTemp_s[output_day_SurfaceTemp_s['Surface']=='WIN_S'],
                  ci='sd',palette='tab10', linewidth=1,legend='full')
ax.set(ylabel='Surface Temperature[C˚]',xlabel='hour', ylim=(0,40))
plt.grid()
#%% 출력데이터 그래프(일사량)
output_day_ir_s = output_day_ir.stack()
output_day_ir_s = output_day_ir_s.reset_index()
output_day_ir_s['month'] = output_day_ir_s.level_0.dt.month
output_day_ir_s['hour'] = output_day_ir_s.level_0.dt.hour

ax = sns.lineplot(x="hour", y=0,hue='level_1', data=output_day_ir_s[output_day_ir_s['level_1']=='Indoor_Beam_Solar'])
ax.set(ylabel='PV Power[kWh]',xlabel='Time', ylim=(0,90), xlim=(1,24))
plt.grid()
plt.legend([])

#%%
df_feature = temp_surface_dc.T

#%% KMeans
from sklearn.cluster import KMeans
df_KMeans = pd.DataFrame(index=df_feature.index)
for i in range(2,8):
    clust_model = KMeans(n_clusters = i)
    clust_model.fit(df_feature)
    centers = clust_model.cluster_centers_
    pred = clust_model.predict(df_feature)
    df_KMeans[str(i)]= pd.Series(pred, index=df_feature.index)
print(df_KMeans)

#%% Cluster result
temp_surface_r = temp_surface.reset_index(drop=True)
cmap = matplotlib.cm.get_cmap('tab10', len(df_KMeans.columns))
color_l = ['r', 'b', 'g', 'grey', 'orange', 'yellow','b']
fig, ax = plt.subplots(2, 3)
for i,c in enumerate(df_KMeans.columns):
    clust = df_KMeans[c]
    temp_surface_r_2 = temp_surface_r[24*100:24*100+24*5]
    for cl, temp in zip(clust, temp_surface_r.columns):
        ax.flat[i].plot(temp_surface_r_2[temp], label=cl, color = cmap(cl), linewidth=1)
        # ax.flat[i].scatter(temp_surface_r_2.index,temp_surface_r_2[temp], label=cl, c = 'white', s=4, ec=cmap(cl))
    ax.flat[i].set(title='Number of surface group: '+ str(c), xlim=(temp_surface_r_2.index[0],temp_surface_r_2.index[-1]))
    # ax.flat[i].grid()
#%% mrt Cal_몬테카를로
c_iter = 70
std_data = 0.5
std_ir = 0
clust_df = df_KMeans.T
k_range= [2,3,4,5,6]
StartTime = datetime.datetime.now()

obj_p = [x_w + L_w / 2, 5, 0.5]
obj_p_c = [x_w + L_w / 2, 1, 0.5]
dataset_list = []
for k in k_range:
    groupBy_df = group_split(k, clust_df) # 군집화
    df_MC_MRT = pd.DataFrame(columns=range(c_iter), index=range(len(temp_surface_dc))) # 빈 df생성
    df_MC_MRT['MRT'] = mrt_real # 참값 저장
    for iter in range(c_iter): # 반복 추출
        input_randomData = random_dataset(1, groupBy_df) # 그룹당 추출하여 파라미터 조합 생성
        df_surface_edited = temp_surface_dc.copy()
        for j in range(k):
            col_grouped = groupBy_df[j] # 그룹당 추출가능 표면온도 집합 생성
            for i in col_grouped:
                df_surface_edited[i] = df_surface_edited[input_randomData[j][0]] # 대표 표면온도로 나머지 덮어쓰기 진행
                for l in df_surface_edited.columns:
                    df_surface_edited[l] = df_surface_edited[l] + np.random.normal(0,std_data,len(df_surface_edited))
        mrt_lins = np.zeros(len(df_surface_edited))
        for d in range(len(df_surface_edited)):
            mrt_lins[d] = cal_mrt_AddSolar(df_surface_edited.iloc[d], output_day['Indoor_Beam_Solar'][d] + np.random.normal(0,std_ir,1)[0],
                                           output_day['Indoor_Diffuse_Solar'][d] + np.random.normal(0,std_ir,1)[0]
                                       ,output_day['Solar_Azimuth'][d], output_day['Solar_Altitude'][d], vf_r, obj_p_c, x_w, z_w, H_w, L_w)[0]
        df_MC_MRT[iter] = mrt_lins
        print(str(k)+': '+str(round(iter/c_iter*100,2)))
    dataset_list.append(df_MC_MRT)
EndtTime = datetime.datetime.now()
print('Running Time: '+str(EndtTime - StartTime))
for i in dataset_list:
    i['MRT'] = mrt_real

df_MC_MRT_total = pd.DataFrame()
for i,df in enumerate(dataset_list):
    df['cluster'] = k_range[i]
    df_MC_MRT_total = pd.concat((df_MC_MRT_total, df))
#%% 분산 분포
df_time = pd.date_range('2021-1-1 01:00:00', '2022-1-1 00:00:00', freq='h')
df_std = pd.DataFrame()
for i in k_range:
    df = df_MC_MRT_total[df_MC_MRT_total['cluster']==i]
    # df_MC_MRT['Time'] = pd.date_range('2022-01-01 01:00', '2023-01-01 00:00', freq='h')
    df_2 = df[df.columns[:c_iter]]
    std_t = (df_2.T).std()
    df_std[i] = std_t
df_std['Time'] = df_time
df_std['hour'] = df_std['Time'].dt.hour
df_std['month'] = df_std['Time'].dt.month
df_std.hist(bins=50)

sns.boxplot(data=df_std, x='month', y=5)
#%% 그래프 - mrt Cal_몬테카를로 표면온도 덮어쓰기

for i,j in enumerate(k_range):
    dataset_n = i +1
    start_point, end_point = 0, -1
    df_MC_MRT_cut = dataset_list[i].iloc[start_point:end_point,:]
    df_MRT_cut = mrt_real[start_point:end_point]
    plt.subplot(len(k_range), 1, dataset_n)
    # df_MC_MRT_cut.plot(color='skyblue', linewidth=0.5, legend=False, alpha=0.5)
    plt.plot(df_MC_MRT_cut, color='skyblue', linewidth=0.5, alpha=0.5)
    plt.plot(dataset_list[0].index[start_point:end_point], df_MRT_cut, color='k',linewidth=0.2)
    plt.ylim(18, 40)
    plt.xlim(df_MC_MRT_cut.index[0],df_MC_MRT_cut.index[-1])
    plt.grid()

#%% 열이름 재설정 / error 계산 - error
col_name = ['MRT']
# for i in range(c_iter):
#     col_name.append('iter_'+str(i))
# for i in range(len(dataset_list)):
#     dataset_list[i].columns = col_name
df_empty = pd.DataFrame()
for j in range(len(dataset_list)):
    error_array = np.zeros(1)
    for i in range(len(mrt_real)):
        error_ = np.array((((dataset_list[j].T)[i][:-1]-mrt_real[i])))
        error_array = np.append(error_array, error_)
    error_s = pd.Series(error_array, name=j+2)
    error_s = error_s[1:]
    df_empty[str(j+2)] = error_s
# df_empty_2 = df_empty[df_empty['2'] != 0]
#%% 오차분포
sns.set_theme(style="darkgrid")
ax = sns.distplot(df_empty['2'])

outpoint = df_empty
#%% Error 그래프 - error  range=[xmin,xmax]
from scipy import stats
x = np.linspace(-1.5,1.5,100)

fig, ax1 = plt.subplots(5, 1)
for i,col in enumerate(df_empty.columns):
    ax1[i].hist(df_empty[col], label=col, bins=30, range=[-1.5,1.5])
    ax1[i].plot(x, stats.norm.pdf(x, loc=0, scale=df_empty[col].std())*len(df_empty[col])/10,
            'k-', lw=1, alpha=0.6, label='norm pdf')
    ax1[i].set_ylim(0,50000)
    CL_error = stats.norm.interval(0.95,loc=0, scale=df_empty[col].std())
    ax1[i].vlines(CL_error[0], 0, 200000, colors='k', linestyles='solid')
    ax1[i].vlines(CL_error[1], 0, 200000, colors='k', linestyles='solid')
dd2 = pd.cut(df_empty[col],30)
# stats.norm.interval(0.95,loc=0, scale=1)
#%%
df_empty = pd.read_csv('./df_error_dist_0921_v2.csv')[['2', '3', '4', '5', '6']]

#%% Error 그래프
df_empty_3 = df_empty.stack()
df_empty_3.name = 'Error'
df_empty_3 = df_empty_3.reset_index()
sns.displot(df_empty_3,bins=200, element='step', hue='level_1', x='Error')
# plt.xlim(-2,2)
#%% Error 그래프
df_empty_4 = df_empty.copy().stack()
df_empty_4 = df_empty_4.reset_index()
index_new = []
for i in range(c_iter*len(k_range)):
    index_new += temp_surface_dc.index
df_empty_4['Time'] = index_new
df_empty_4['month'] = df_empty_4['Time'].dt.month
df_empty_4['hour'] = df_empty_4['Time'].dt.hour

sns.boxplot(x="hour", y=0, hue='level_1', data=df_empty_4)
sns.boxplot(x="hour", y=0, data=df_empty_4)
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
