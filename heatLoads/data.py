import pandas as pd
import numpy as np
import streamlit as st

from heatLoads.model import * 

@st.cache
def simulation(dzo, dwl, dwd, dwg):
    
    dzo = dzo.fillna(0)
    dzo = dzo.set_index('walls')
    grdf = dzo[2:].T.groupby(dzo.T.index)

    dwl = dwl.fillna(0)
    dwl = dwl.set_index('parameters')

    dwd = dwd.fillna(0)
    dwd = dwd.set_index('parameters')

    dwg = dwg.fillna(0)
    dwg = dwg.set_index('parameters')
    dwg = dwg.T

    # Orientation clockwise from South
    azimuth_12h = dwg['azimuth_12h_deg'].values[0]
    h_ceiling   = dwg['h_ceiling_m'].values[0]

    # Infiltration rate
    n_inf = dwg['n_infiltr_per_h'].values[0]

    # Internal blind : 1 = True, 0 = False
    internal_blind  = dwg['internal_blind'].values[0]

    # Wall thermal mass accessibility from internal side: 1 = True, 0 = False
    suspended_ceiling = dwg['suspended_ceiling'].values[0]
    raised_floor      = dwg['raised_floor'].values[0]
    carpet            = dwg['carpet'].values[0]

    # Month comprised between 1 and 12
    month = int(dwg['month'].values[0])

    # Indoor temperature set point, in °C
    t_in  = dwg['t_in_C'].values[0]

    # Temperature of adjacent spaces, in °C
    t_adj = dwg['t_adj_C'].values[0]

    # Daily Plant operating hours comprised between 0 and 24 h (0 to 24 h for continuous operating system)
    hour_start_plant  = dwg['plant_start_h'].values[0]
    hour_stop_plant   = dwg['plant_stop_h'].values[0] 

    # Daily occupancy hours comprised between 0 and 24 h
    hour_start_occ    = dwg['occ_start_h'].values[0]
    hour_stop_occ     = dwg['occ_stop_h'].values[0]

    # Occupants activity
    q_radi_occ   = dwg['occ_rad_W'].values[0]
    q_conv_occ   = dwg['occ_conv_W'].values[0]

    # Appliances and Lighting specific heat gains
    q_conv_appli = dwg['appliances_Wm2'].values[0]
    q_radi_light = dwg['lighting_rad_Wm2'].values[0]
    q_conv_light = dwg['lighting_conv_Wm2'].values[0]

    # Reduction of load due to absorption in the return air stream
    f_light = dwg['lighting_factor'].values[0]
    f_roof  = dwg['roof_factor'].values[0]

    # External walls short wave absorption factors
    alpha_wall = dwg['alpha_wall'].values[0]
    alpha_roof = dwg['alpha_roof'].values[0]

    # Not used
    op_wall_shading = dwg['opaque_wall_shading'].values[0]

    n_day_sim=3
    n_hours_sim=24*n_day_sim
    DELTAtau=600 * 2
    nsph = int(3600/DELTAtau)
    n_steps = int(1 + n_hours_sim * nsph)

    hour_start=0
    hour_stop=hour_start + n_hours_sim

    tau_initial=hour_start*3600
    tau_final=hour_stop*3600
    tau = np.arange(tau_initial,tau_final+1,DELTAtau)

    hour       = tau/3600
    hour_per_0 = hour-24*np.trunc(hour/24)
    # np.choose(condition,[action if condition = 0 or false, action if condition = 1 or true])
    # np.choose(array, [action if condition = 0, action if condition = 1 , action if condition = 2 ...)])
    hour_per=np.choose(hour_per_0 > 0.000001,[24,hour_per_0])

    t_out_1d = np.interp(DELTAtau * np.arange(24*nsph), 3600 * np.arange(25), tout(month))
    t_out = t_out_1d
    for n in range(n_day_sim-1):
        t_out = np.append(t_out, t_out_1d)
    t_out = np.append(t_out, t_out_1d[0])

    names = dzo.columns.values
    nocc  = dzo.T.n_occ.values
    arfl  = dzo.T.floor_m2.values

    Q_r_gains = q_radi_occ * nocc +  f_light * q_radi_light * arfl
    Q_c_gains = q_conv_occ * nocc + (f_light * q_conv_light + q_conv_appli)* arfl

    dicaz = {'ori': ['12h','3h','6h','9h'], \
             'azimuth': [azimuth_norm(azimuth_12h), azimuth_norm(azimuth_12h + 90), \
                        azimuth_norm(azimuth_12h + 180), azimuth_norm(azimuth_12h + 270)]}
    daz = pd.DataFrame(data=dicaz)
    daz = daz.set_index('ori')

    M_A = np.zeros(len(names))
    dfz = dict()
    for i in range(len(names)):
        zn = names[i]
        dft  = grdf.get_group(zn).copy()
        dft  = dft.rename(columns=lambda x: x.replace('_m2', ''), index={zn: 'A'})
        dftt = dft.T.copy()
        dftt['ori']   = (dftt.index.str.split("_").str[0]).where(dftt.index.str.contains('_'), '0h')
        dftt['wtype'] = (dftt.index.str.split("_").str[1]).where(dftt.index.str.contains('_'), dftt.index.str.split("_").str[0])
        dftt = pd.merge(dftt, daz, left_on='ori', right_index=True, how= 'outer')
        dftt = pd.merge(dftt, dwl.T, left_on='wtype', right_index=True, how= 'outer')
        dftt = pd.merge(dftt, dwd.T, left_on='wtype', right_index=True, how= 'outer')
        dftt['U_Wm2K_x'].update(dftt['U_Wm2K_y'])
        dftt = dftt.rename(columns={"U_Wm2K_x": "U_Wm2K"})
        dftt = dftt.drop(columns=['U_Wm2K_y', 'H_B', 'A1_deg', 'A2_deg', 'A3_deg', 'D_H'])
        dftt = dftt.fillna(0)
        dftt['Mi_kg'] = dftt['A'] * dftt['mi_kgm2']
        dftt['Mt_kg'] = dftt['A'] * dftt['mt_kgm2']
        dftt['AU']    = dftt['A'] * dftt['U_Wm2K']
        dftt['Afg']   = dftt['A'] * dftt['g_gl'] * (1 - dftt['f_fr'])
        dfz[zn] = dftt.copy()
        condmi = (dftt.index.str.contains('ex')) |  (dftt.index.str.contains('roof'))
        condmt = (dftt.index.str.contains('ad')) |  (dftt.index.str.contains('in'))
        M = dftt[condmi]['Mi_kg'].sum() + dftt[condmt]['Mt_kg'].sum() / 2
        M_A[i] = M / arfl[i] if arfl[i] > 0 else 0

    condex = (dftt.index.str.contains('ex')) | (dftt.index == 'roof')
    listaz = dftt[condex]['azimuth'].unique()
    condex = (dwl.T.index.str.contains('ex')) | (dwl.T.index == 'roof')
    listwl = dwl.T[condex].index.unique().tolist()

    dictwlazDTE = dict()
    for k in range(len(listwl)):
        wl = listwl[k]
        M_A_w = dwl.T.loc[wl, 'mt_kgm2']
        if (wl == 'roof') :
            az = 180
            sl = 0
            dte = DTE(alpha_roof, M_A_w, az, sl, op_wall_shading, month, t_in)  
            for l in range(len(listaz)):
                az = listaz[l]
                dictwlazDTE[(wl,az)] = f_roof * dte    
        else:
            for l in range(len(listaz)):
                az = listaz[l]
                sl = 90
                dte = DTE(alpha_wall, M_A_w, az, sl, op_wall_shading, month, t_in)
                dictwlazDTE[(wl,az)] = dte

    condwd = (dftt.index.str.contains('wd'))
    listaz = dftt[condwd]['azimuth'].unique()
    listwd = dwd.T.index.unique().tolist()
    dictwdazIs = dict()
    dictwdazIm = dict()

    for k in range(len(listwd)):
        wd = listwd[k]
        for l in range(len(listaz)):
            az = listaz[l]
            H_B = dwd.T.loc[wd, 'H_B']
            A1  = dwd.T.loc[wd, 'A1_deg']
            A2  = dwd.T.loc[wd, 'A2_deg']
            A3  = dwd.T.loc[wd, 'A3_deg']
            D_H = dwd.T.loc[wd, 'D_H']
            Ishade, Inoshade = Iwd(az, H_B, A1, A2, A3, D_H, month)
            dictwdazIs[(wd,az)] = Ishade
            dictwdazIm[(wd,az)] = Inoshade.max() 

    # plt.plot(dictwdazIs[('wd1', 180.0)])

    df = pd.DataFrame()

    for i in range(len(names)):

        zn = names[i]

        dfzt   = dfz[zn].copy()

        condad = (dfzt.index.str.contains('ad')) 
        dfzad  = dfzt[condad]
        Qadtr  = np.zeros(n_steps)
        for k in range(len(dfzad)):
            if dfzad['A'].values[k] > 0 :
                Qadtr = Qadtr + dfzad['AU'].values[k] * (t_adj - t_in)

        condex = (dfzt.index.str.contains('ex')) | (dfzt.index == 'roof')
        dfzex  = dfzt[condex]
        Qdte = np.zeros(n_steps)
        for k in range(len(dfzex)):
            if dfzex['A'].values[k] > 0 :
                wl = dfzex['wtype'].values[k]
                az = dfzex['azimuth'].values[k]
                Qdte = Qdte + dfzex['AU'].values[k] * dictwlazDTE[(wl,az)]

        condwd = (dfzt.index.str.contains('wd'))
        dfzwd  = dfzt[condwd]
        Qwdtr = np.zeros(n_steps)
        Qsol = np.zeros(n_steps)
        Qsolmax = 0
        H_B = 1
        for k in range(len(dfzwd)):
            if dfzwd['A'].values[k] > 0 :
                wd = dfzwd['wtype'].values[k]
                az = dfzwd['azimuth'].values[k]
                Qwdtr   = Qwdtr   + dfzwd['AU'].values[k]  * (t_out - t_in)
                Qsol    = Qsol    + dfzwd['Afg'].values[k] * dictwdazIs[(wd,az)]
                Qsolmax = Qsolmax + dfzwd['Afg'].values[k] * dictwdazIm[(wd,az)]

        # Windows solar gains, convective and radiative solar gains      
        Qint = Qdot(internal_blind, suspended_ceiling, raised_floor, carpet, \
                Q_r_gains[i], Q_c_gains[i], hour_start_occ, hour_stop_occ,   \
                  hour_start_plant, hour_stop_plant, M_A[i], H_B, Qsolmax, Qsol, month)

        Q_dot_cooling = Qadtr + Qdte + Qwdtr + Qint

        df[zn + ' (W)'] = Q_dot_cooling

    df["Total (W)"] = df.sum(axis=1)
    df = df.astype(int)
    df['hour']= hour
    df['hour_per']= hour_per
    df['inthour']= df['hour'].astype(int)
    df = df.loc[df['hour'] == df['inthour']]
    df = df.drop(['hour', 'inthour'], axis=1)
    df = df.reset_index()
    df = df.loc[n_hours_sim-24:n_hours_sim]
    df = df.drop(['index'], axis=1)
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    df = df.copy()
    df['hour_per'] = df['hour_per'].astype(int)
    df['hour_per'][0] = 0
    df = df.rename(columns={'hour_per': 'Hour (solar time)'})
    df.index = df['Hour (solar time)']
    df = df.drop(['Hour (solar time)'], axis=1)
        
    results = df
        
    return results