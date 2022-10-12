import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pvlib as pv
import random


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
def cal_VewFactor_point(point, span_list, N_split):
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
def cal_mrt_c(temp_pr):
    mrt = (0.18*(temp_pr[2]+temp_pr[5])+0.22*(temp_pr[0]+temp_pr[3])+0.30*(temp_pr[1]+temp_pr[4]))/(2*0.7)
    return mrt
def cal_q_long_wave(vf_r, temp):
    q_long_wave = np.zeros((len(vf_r.columns)))
    for i in range(len(vf_r.columns)):
        q_long_wave[i] = (vf_r[vf_r.columns[i]] * ((temp + 273) ** 4)).sum()
    return q_long_wave
def cal_q_sol(ir_dir, ir_dff, sol_azimuth, sol_altitude, A_window, vf_r):
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
def output_open(output_csv, list_name_result,list_name_new, time_start, time_end):
    df = pd.read_csv(output_csv)[list_name_result]
    df_result = df.rename(columns=dict(zip(list_name_result, list_name_new)))
    df_result.index = pd.date_range(time_start, time_end, freq='1H')
    df_result['Indoor Solar Irradiation[W/m²]'] = df_result['Indoor_Beam_Solar'] + df_result['Indoor_Diffuse_Solar']
    df_result['hour'] = df_result.index.hour
    df_result['month'] = df_result.index.month
    return df_result
def Check_GeoBC(span_list, FLOOR, N_split):
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
def cal_mrt_AddSolar(Temp_surface, solar_beam, solar_diffuse, Solar_Azimuth, Solar_Altitude,
                     vf, obj_p, x_w, z_w, H_w, L_w):
    T_sur = Temp_surface.values
    sol_rad = cal_q_sol(solar_beam, solar_diffuse, Solar_Azimuth, Solar_Altitude, 1, vf)
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
def cal_MRT(vf_r, temp_surface, output, distance_from_Win,obj_p_c,x_w, z_w, H_w, L_w):
    df_mrt_real = pd.DataFrame()
    df_c_shade = pd.DataFrame()
    for i in [distance_from_Win]:
        mrt_real = np.linspace(0,0,len(temp_surface))
        c_shade = np.linspace(0,0,len(temp_surface))
        for d in range(len(temp_surface)):
            mrt_sol_c = cal_mrt_AddSolar(temp_surface.iloc[d], output['Indoor_Beam_Solar'][d],
                                           output['Indoor_Diffuse_Solar'][d]
                                           , output['Solar_Azimuth'][d],
                                         output['Solar_Altitude'][d],
                                         vf_r, obj_p_c,x_w, z_w, H_w, L_w)
            mrt_real[d] = mrt_sol_c[0]
            c_shade[d] = mrt_sol_c[2]
            print(str(i)+'m:'+str(round(d/len(temp_surface),2)))
        df_mrt_real[str(i)+'m'] = mrt_real
        df_c_shade[str(i)+'m'] = c_shade
    df_mrt_real.index = temp_surface.index
    df_c_shade.index = temp_surface.index
    return df_mrt_real, df_c_shade
