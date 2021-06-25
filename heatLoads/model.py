import pandas as pd                 # Data tables
import numpy as np                  # Arrays

from math import sqrt, atan, log, exp, sin, cos, tan

from scipy.integrate import odeint
from scipy.optimize import *

pi = np.pi

month = 7

# "!Boundary layers"
h_r=5 # [W/m^2-K]
h_c = 7 # [W/m^2-K]
h_in=h_r + h_c

# "Air properties:"
v_a=0.8401 # [m^3/kg] "specific volume of humid air per kg of dry air"
c_p_a=1020 # [J/kg-K] "specific heat capacity of humid air per kg of dry air"
sigma_boltzman=5.67E-8

# "! Room Data"
# "Room height"
Height_room=2.7 # [m]
# "Lenght on 6h-0h direction"
Lenght_6h_0h=5.4 # [m]
# "Lenght on 9h-3h direction"
Lenght_9h_3h=1.8 # [m]

# "! Windows areas supposed to be included in 0h wall"
Height_wd_0 =1.2 # [m]
Breadth_wd_0 =1.6 # [m]
Height_wd_sill_0 =0.8 #[m]

thickness_gl = 0.006 #[m]

# "! Windows parameters"
U_wd=1.49 # [W/m^2-K]
SF_gl=0.6
f_frame=0.3
rho_blind=0.64
rho_ground=0
slope_deg=90

# glazing
rho_glazing=2500 #[kg/m^3]
lambda_glazing=1.0 #[W/m.K]
c_p_glazing=750 #[J/kg.K]

# concrete bloc
rho_concrete_bloc=1200 #[kg/m^3]
lambda_concrete_bloc=1.273 #[W/m.K]
c_p_concrete_bloc=840 #[J/kg.K]

# Hollow concrete 
rho_hollow_concrete=1600 #[kg/m^3]
lambda_hollow_concrete=1.182 #[W/m.K]
c_p_hollow_concrete=840 #[J/kg.K]

# plaster
e_suspended_ceiling=0.01 #[m]
lambda_suspended_ceiling=0.2 #[W/m.K]
rho_suspended_ceiling=1300 #[kg/m^3]
c_p_suspended_ceiling=840 #[J/kg.K]

# wood panel
e_raised_floor=0.02 #[m]
lambda_raised_floor=0.2 #[W/m.K]
rho_raised_floor=1600 #[kg/m^3]
c_p_raised_floor=800 #[J/kg.K]

# carpet
e_carpet=0.02 #[m]
lambda_carpet=0.2 #[W/m.K]
rho_carpet=1600 #[kg/m^3]
c_p_carpet=800 #[J/kg.K]

# blind
e_blind=0.002 #[m]
lambda_blind=0.2 #[W/m.K]
rho_blind=1600 #[kg/m^3]
c_p_blind=800 #[J/kg.K]

U_half_carpet = 2 * lambda_carpet/e_carpet

# Air layer
R_air_layer=0.17 #[m^2.K/W] #"0.17 for air +0.18 for insulation#"
U_air_layer=1/R_air_layer

thickness_air_layer=0.06#[m]
rho_air_layer=1.2 #[kg/m^3]
lambda_air_layer=thickness_air_layer/R_air_layer
c_p_air_layer=1060 #[J/kg.K]



def Qdot(iflag_internal_blind, iflag_suspended_ceiling, iflag_raised_floor, iflag_carpet, \
         Q_dot_rad_max, Q_dot_conv_max, hour_start_occ, hour_stop_occ,   \
         hour_start_plant, hour_stop_plant, M_A, H_B, Q_dot_sol_wd_max_no_shading, \
         Q_dot_sol_wd, month): 
    
    # "!Boundary layers"
    h_r=5 # [W/m^2-K]
    h_c = 7 # [W/m^2-K]
    h_in=h_r + h_c
    
    # "! Room Data"
    # "Room height"
    Height_room=2.7 # [m]
    # "Lenght on 6h-0h direction"
    Lenght_6h_0h=5.4 # [m]
    # "Lenght on 9h-3h direction"
    Lenght_9h_3h=1.8 # [m]

    # "! Windows areas supposed to be included in 0h wall"
    Height_wd_0 =1.2 # [m]
    Breadth_wd_0 =1.6 # [m]
    Height_wd_sill_0 =0.8 #[m]

    H_B_wd=max(0.001,H_B)
    area_wd = Height_wd_0 * Breadth_wd_0
    Height_wd = (area_wd * H_B_wd) ** 0.5
    Breadth_wd = (area_wd / H_B_wd) ** 0.5
    Height_wd_sill = max(0.1, min(Height_wd_sill_0, Height_room - Height_wd) )
    
    area_wall_0h=max(0,Height_room*Lenght_9h_3h-area_wd)
    area_wall_3h=Height_room*Lenght_6h_0h
    area_wall_6h=Height_room*Lenght_9h_3h
    area_wall_9h=Height_room*Lenght_6h_0h
    area_ceiling=Lenght_9h_3h*Lenght_6h_0h
    area_floor=Lenght_9h_3h*Lenght_6h_0h

    n_walls=6
    area_wall = np.array([area_wall_0h, area_wall_3h, area_wall_6h, area_wall_9h, area_ceiling, area_floor])
    area_wall_wd=np.sum(area_wall)+area_wd

    #"! Estimated Floor Structure Mass and Wall Mass per square meter of area"
    M_per_A_tot=max(10,M_A)
    M_tot=max(100,area_floor*M_per_A_tot)
    M_per_A_wall = M_tot/((area_wall[0]+np.sum(area_wall[1:4])/2)+2*(area_ceiling/2+area_floor/2))
    M_per_A_floor=2*M_per_A_wall

    #!Indoor air capacity"
    C_a_in=5*Height_room*Lenght_6h_0h*Lenght_9h_3h*c_p_a/v_a

    # Glazing capacity
    C_gl = area_wd * (1-f_frame) * thickness_gl * rho_glazing * c_p_glazing

    # Suspended ceiling
    C_A_ce = e_suspended_ceiling*rho_suspended_ceiling*c_p_suspended_ceiling

    # Raised floor
    C_A_fl = e_raised_floor*rho_raised_floor*c_p_raised_floor

    # Carpet
    C_A_cp = e_carpet*rho_carpet*c_p_carpet

    # Blind
    C_bl = area_wd * e_blind*rho_blind*c_p_blind

    # "!Total number of finite element layers, with two degree two elements by layer"
    n_layers = 2
    nl=n_layers

    # "! internal vertical wall layers"
    thickness_wall= (M_per_A_wall/2)/rho_concrete_bloc
    thickness_wall_int= thickness_wall/n_layers * np.ones(n_layers) 
    lambda_wall_int=  lambda_concrete_bloc * np.ones(n_layers)
    rho_wall_int=  rho_concrete_bloc * np.ones(n_layers)
    c_layer_wall_int= c_p_concrete_bloc * np.ones(n_layers)

    # "! floor layers"
    thickness_floor=(M_per_A_floor/2)/rho_hollow_concrete
    thickness_floor_int= thickness_floor/n_layers * np.ones(n_layers)
    lambda_floor_int=  lambda_hollow_concrete * np.ones(n_layers)
    rho_floor_int=  rho_hollow_concrete * np.ones(n_layers)
    c_layer_floor_int= c_p_hollow_concrete * np.ones(n_layers)

    # Reverse arrays
    thickness_floor_int_2 =thickness_floor_int[::-1]
    lambda_floor_int_2 =lambda_floor_int[::-1]
    rho_floor_int_2 =rho_floor_int[::-1]
    c_layer_floor_int_2 =c_layer_floor_int[::-1]

    # Matrixes of vertical wall layers"
    n_elem,R_nobl_wall,L_wall,C_wall = wall_matrix(n_layers,thickness_wall_int,lambda_wall_int,rho_wall_int,c_layer_wall_int)

    # Matrixes of floor layers"
    n_elem,R_nobl_floor,L_floor,C_floor = wall_matrix(n_layers,thickness_floor_int,lambda_floor_int,rho_floor_int,c_layer_floor_int)

    n_nodes=2*n_elem+1

    shape = (n_walls, n_nodes, n_nodes)
    C_matrix = np.zeros(shape) ; L_matrix = np.zeros(shape)

    # External wall
    C_matrix[0] = C_wall; L_matrix[0] = L_wall
    # Internal walls
    C_matrix[1] = C_wall; L_matrix[1] = L_wall
    C_matrix[2] = C_wall; L_matrix[2] = L_wall
    C_matrix[3] = C_wall; L_matrix[3] = L_wall
    # Ceiling
    C_matrix[4] = C_wall; L_matrix[4] = L_wall
    # Floor
    C_matrix[5] = C_floor; L_matrix[5] = L_floor

    # "! ComputeView factor from walls to walls, from walls to window and from window to walls"
    FV_wall,FV_to_wd,FV_wd_to = View_factors(Lenght_6h_0h,Lenght_9h_3h,Height_room,Breadth_wd,Height_wd_sill,Height_wd)

    # Boundary layers"
    Ah_c_wall=h_c*area_wall
    Ah_r_wall_to_wd=h_r*area_wall*FV_to_wd
    Ah_r_wd_to_wall=h_r*area_wd*FV_wd_to
    Ah_r_wall=h_r*np.diag(area_wall) @ FV_wall

    Ah_c_wd=area_wd*h_c
    Ah_c_internal_blind=2*area_wd*h_c
    Ah_r_wd_internal_blind=area_wd*h_r

    #!Window"
    R_wd_no_in_bl=max(0,1/U_wd-1/h_in)
    U_wd_no_in_bl=1/R_wd_no_in_bl
    AU_wd_no_in_bl=area_wd*U_wd_no_in_bl

    if iflag_raised_floor==1 and iflag_carpet==1:
        iflag_carpet=0

    hour_start_coolingplant=min(24,max(0,hour_start_plant))
    hour_stop_coolingplant=min(24,max(0,hour_stop_plant))
    hour_start_occupancy=min(24,max(0,hour_start_occ))
    hour_stop_occupancy=min(24,max(0,hour_stop_occ))
    
    # "!Set points"
    t_out=26 #[°C]
    t_a_in_set=26 #[°C]
    t_init=26 #[°C]
    C_t_in=2 #[K^-1]

    # Initial conditions"
    t_a_in_init=t_a_in_set
    U_c_in_init=C_a_in*t_init
    

    # "! Cooling system sizing"
    Q_dot_cooling_max=Q_dot_sol_wd_max_no_shading + Q_dot_rad_max + Q_dot_conv_max
    

    # Simulation period
    n_day_sim=3
    hour_start=0
    n_hours_sim=24*n_day_sim
    hour_stop=hour_start + n_hours_sim

    tau_initial=hour_start*3600
    tau_final=hour_stop*3600

    DELTAtau=600  * 2 #[s]

    # Time in s : Create an array of evenly-spaced values
    tau = np.arange(tau_initial,tau_final+1,DELTAtau)

    # Hour and Day from the start of the simulation
    hour       = tau/3600
    hour_per_0 = hour-24*np.trunc(hour/24)
    # np.choose(condition,[action if condition = 0 or false, action if condition = 1 or true])
    # np.choose(array, [action if condition = 0, action if condition = 1 , action if condition = 2 ...)])
    hour_per=np.choose(hour_per_0 > 0.000001,[24,hour_per_0])

    day = hour/24
    day_int_0 = np.trunc(hour/24)+1
    day_int   = day_int_0-1

    # Sarting hour in sun data table according to month of simulation
    month_start=max(1,min(12,month))
    hour_start = np.zeros(13)
    hour_start[1]=1+15*24; hour_start[2]=1+(31+15)*24; hour_start[3]=1+(31+28+15)*24; hour_start[4]=1+(2*31+28+15)*24; hour_start[5]=1+(2*31+28+30+15)*24; hour_start[6]=1+(3*31+28+30+15)*24
    hour_start[7]=1+(3*31+28+2*30+15)*24; hour_start[8]=1+(4*31+28+2*30+15)*24; hour_start[9]=1+(5*31+28+2*30+15)*24; hour_start[10]=1+(5*31+28+3*30+15)*24
    hour_start[11]=1+(6*31+28+3*30+15)*24; hour_start[12]=1+(6*31+28+4*30+15)*24

    # Hour and Day from the start of the year (simulation starts at 15th of the considered month)
    hour_yr = hour + float(hour_start[month]) - 1
    day_yr = hour_yr/24
    
    
#     #Gains factor
#     f_gains = np.where(Q_dot_sol_wd_max_no_shading > 0.1, Q_dot_sol_wd_no_shading/Q_dot_sol_wd_max_no_shading, 0)

    # Plant on/off
    #Smooth starting of the system when intermittent cooling is performed
#     hour_start_occupancy = hour_start_coolingplant
    f_plant=np.asarray([plant(hi,hour_start_coolingplant,hour_stop_coolingplant,hour_start_occupancy) for hi in hour_per])
    
    #!Occupancy heat gains#
    f_gains_occ=np.asarray([occupancy(hi,hour_start_occupancy,hour_stop_occupancy) for hi in hour_per])

    # Heat gains
    #! Share internal heat gains into radiative and convective#
    Q_dot_sensible_gains_rad=f_gains_occ*Q_dot_rad_max   
    Q_dot_sensible_gains_conv=f_gains_occ*Q_dot_conv_max     

    #! Radiative sensible heat gains shared as function of walls areas#

    if area_wall_wd>0:
        Q_dot_rad_wall= np.asarray([np.asarray([x*y/area_wall_wd  for y in Q_dot_sensible_gains_rad]) for x in area_wall])
        Q_dot_rad_wd=Q_dot_sensible_gains_rad*area_wd/area_wall_wd
    else:
        Q_dot_rad_wall= np.asarray([np.asarray([x*y*0  for y in Q_dot_sensible_gains_rad]) for x in area_wall])
        Q_dot_rad_wd= Q_dot_sensible_gains_rad*0
        

    # Radiative sensible solar heat gains on floor except if there is an internal blind
#     Q_dot_rad_wall= np.zeros((n_walls, len(tau)))
#     Q_dot_rad_wd= np.zeros((len(tau)))

    Q_dot_rad_wall[5,:] = Q_dot_rad_wall[5,:] + (1-iflag_internal_blind)*Q_dot_sol_wd

    C_t_in = 2

    tau_0  = tau[0]

    t_a_in_0 = t_init
    t_s_wd_0 = t_init
    t_s_ce_0 = t_init
    t_s_fl_0 = t_init
    t_s_cp_0 = t_init
    t_s_bl_0 = t_init

    Ti0  = [t_a_in_0 ]
    Ti0.extend([t_s_wd_0 ])
    Ti0.extend([t_s_ce_0 ])
    Ti0.extend([t_s_fl_0 ])
    Ti0.extend([t_s_cp_0 ])
    Ti0.extend([t_s_bl_0 ])
    Ti0.extend(t_init * np.ones(n_nodes * n_walls))

    Q_dot_cooling_v = []

    t_a_in_set = 26.01


    def model_dTi_t(Ti, tau):
        T1 = Ti[0]
        T2 = Ti[1]
        T3 = Ti[2]
        T4 = Ti[3]
        T5 = Ti[4]
        T6 = Ti[5]
        Tw = Ti[6:].reshape((n_walls, n_nodes))

        ind = int((tau - tau_0) /DELTAtau) 
        if ind>len(f_plant)-1: ind=len(f_plant)-1

        # Internal air capacity heat balance#
        t_a_in = T1

        if (C_t_in*(t_a_in-t_a_in_set) > 0 ) and (C_t_in*(t_a_in-t_a_in_set) < 1 ) :
            X_cooling=C_t_in*(t_a_in-t_a_in_set)
        elif C_t_in*(t_a_in-t_a_in_set) > 1 :
            X_cooling=1
        else:
            X_cooling=0

        Q_dot_cooling=f_plant[ind]*X_cooling*Q_dot_cooling_max
        Q_dot_cooling_v.append(tau)

        # Glazing temperature
        t_s_wd = T2

        # Suspended ceiling temperature
        t_s_ce = T3

        # Raised floor temperature
        t_s_fl = T4

        # Carpet temperature
        t_s_cp = T5

        # Internal blind temperature
        t_s_bl = T6

        # Radiative and convective heat exchanges between wall surfaces
        t_s_wall = Tw[:,0]

        if iflag_suspended_ceiling == 1 : t_s_wall[4] = t_s_ce
        if iflag_raised_floor == 1      : t_s_wall[5] = t_s_fl
        if iflag_carpet == 1            : t_s_wall[5] = t_s_cp

        # Wall surface nodes#
        Q_dot_r_i_to_j = np.diag(t_s_wall) @ Ah_r_wall - Ah_r_wall @ np.diag(t_s_wall)

        # Wall to indoor air convective exchanges#
        Q_dot_c_in_to_wall = Ah_c_wall*(t_a_in-t_s_wall)

        # Window heat balance
        Q_dot_out_to_wd= AU_wd_no_in_bl*(t_out-t_s_wd)
        Q_dot_c_wd_to_in=Ah_c_wd*(t_s_wd-t_a_in)
        Q_dot_r_bl_to_wd=iflag_internal_blind*area_wd*sigma_boltzman*(t_s_bl+273)**4
        Q_dot_r_wd_to_bl=iflag_internal_blind*area_wd*sigma_boltzman*(t_s_wd+273)**4

        # Internal blind heat balance
        Q_dot_c_bl_to_in=2*area_wd*h_c*(t_s_bl-t_a_in)

        # wall to window radiative exchanges if there is no internal blind
        Q_dot_r_wall_to_wd = (1-iflag_internal_blind)*Ah_r_wall_to_wd/h_r* sigma_boltzman*(t_s_wall+273)**4 * (t_s_wall-t_s_wd)
        Q_dot_r_wd_to_wall = (1-iflag_internal_blind)*Ah_r_wd_to_wall/h_r* sigma_boltzman*(t_s_wd+273)**4 * (t_s_wd-t_s_wall)

        # wall to internal blind radiative exchanges if there is an internal blind
        Q_dot_r_wall_to_bl = iflag_internal_blind*Ah_r_wall_to_wd/h_r* sigma_boltzman*(t_s_wall+273)**4 * (t_s_wall-t_s_wd)
        Q_dot_r_bl_to_wall = iflag_internal_blind*Ah_r_wd_to_wall/h_r* sigma_boltzman*(t_s_bl+273)**4 * (t_s_bl-t_s_wall)

        # Wall surface node heat balance; Matrix Aij with axis=0 > sum on first index i i.e. sum of each column
        Q_dot_in_to_wall = Q_dot_r_wd_to_wall - Q_dot_r_wall_to_wd + Q_dot_rad_wall[:,ind] + Q_dot_c_in_to_wall+ \
        np.sum(Q_dot_r_i_to_j, axis=0) - np.sum(Q_dot_r_i_to_j, axis=1) + Q_dot_r_bl_to_wall - Q_dot_r_wall_to_bl

        i1 = -Q_dot_cooling-np.sum(Q_dot_c_in_to_wall)+Q_dot_c_wd_to_in+Q_dot_c_bl_to_in+Q_dot_sensible_gains_conv[ind]
        C1 = C_a_in
        dT1_t =  i1/C1

        i2 = Q_dot_out_to_wd+np.sum(Q_dot_r_wall_to_wd)+Q_dot_rad_wd[ind]-np.sum(Q_dot_r_wd_to_wall)-Q_dot_c_wd_to_in+ \
             Q_dot_r_bl_to_wd-Q_dot_r_wd_to_bl
        C2 = C_gl
        dT2_t =  i2/C2

        i3 = Q_dot_in_to_wall[4]/area_wall[4]
        C3 = C_A_ce
        dT3_t =  i3/C3

        i4 = Q_dot_in_to_wall[5]/area_wall[5]
        C4 = C_A_fl
        dT4_t =  i4/C4

        i5 = Q_dot_in_to_wall[5]/area_wall[5]
        C5 = C_A_cp
        dT5_t =  i5/C5

        i6 = iflag_internal_blind*Q_dot_sol_wd[ind]+Q_dot_r_wd_to_bl-Q_dot_r_bl_to_wd +np.sum(Q_dot_r_wall_to_bl) \
             -np.sum(Q_dot_r_bl_to_wall)-Q_dot_c_bl_to_in
        C6 = C_bl
        dT6_t =  i6/C6

        #! All walls
        shape = (n_walls, n_nodes)
        fh_ext = np.zeros(shape)
        dTw_t = np.zeros(shape)

        fh_ext[:,0]= Q_dot_in_to_wall/area_wall

        if iflag_suspended_ceiling == 1 : fh_ext[4,0] = U_air_layer   * (t_s_wall[4] - Tw[4,0])
        if iflag_raised_floor == 1      : fh_ext[5,0] = U_air_layer   * (t_s_wall[5] - Tw[5,0])
        if iflag_carpet == 1            : fh_ext[5,0] = U_half_carpet * (t_s_wall[5] - Tw[5,0])

        for i in range(n_walls):
            dTw_t[i,:] = np.linalg.inv(C_matrix[i,:,:]) @ (fh_ext[i,:] - L_matrix[i,:,:] @ Tw[i,:])

        dTi_t = [dT1_t]
        dTi_t.extend([dT2_t])
        dTi_t.extend([dT3_t])
        dTi_t.extend([dT4_t])
        dTi_t.extend([dT5_t])
        dTi_t.extend([dT6_t])
        dTi_t.extend(dTw_t.flatten())

        return (dTi_t, Q_dot_cooling) # allows to return more outputs than only dTi_T

    def dTi_t(Ti, tau):
        ret = model_dTi_t(Ti, tau) # only dTi_T is returned for ordinary differential integral
        return ret[0]

    Ti = odeint(dTi_t , Ti0, tau)

    # dTi_t(np.array(Ti0), 600)

    Q_dot_cooling = np.asarray([model_dTi_t(Ti[tt],tau[tt])[1] for tt in range(len(tau))])


    return Q_dot_cooling






def Isol(month):
    # Simulation period
    n_day_sim=3
    hour_start=0
    n_hours_sim=24*n_day_sim
    hour_stop=hour_start + n_hours_sim

    tau_initial=hour_start*3600
    tau_final=hour_stop*3600

    DELTAtau=600 * 2  #[s]

    # Time in s : Create an array of evenly-spaced values
    tau = np.arange(tau_initial,tau_final+1,DELTAtau)

    # Hour and Day from the start of the simulation
    hour       = tau/3600
    hour_per_0 = hour-24*np.trunc(hour/24)
    # np.choose(condition,[action if condition = 0 or false, action if condition = 1 or true])
    # np.choose(array, [action if condition = 0, action if condition = 1 , action if condition = 2 ...)])
    hour_per=np.choose(hour_per_0 > 0.000001,[24,hour_per_0])

    day = hour/24
    day_int_0 = np.trunc(hour/24)+1
    day_int   = day_int_0-1

    # Sarting hour in sun data table according to month of simulation
    month_start=max(1,min(12,month))
    hour_start = np.zeros(13)
    hour_start[1]=1+15*24; hour_start[2]=1+(31+15)*24; hour_start[3]=1+(31+28+15)*24; hour_start[4]=1+(2*31+28+15)*24; hour_start[5]=1+(2*31+28+30+15)*24; hour_start[6]=1+(3*31+28+30+15)*24
    hour_start[7]=1+(3*31+28+2*30+15)*24; hour_start[8]=1+(4*31+28+2*30+15)*24; hour_start[9]=1+(5*31+28+2*30+15)*24; hour_start[10]=1+(5*31+28+3*30+15)*24
    hour_start[11]=1+(6*31+28+3*30+15)*24; hour_start[12]=1+(6*31+28+4*30+15)*24

    # Hour and Day from the start of the year (simulation starts at 15th of the considered month)
    hour_yr = hour + float(hour_start[month]) - 1
    day_yr = hour_yr/24

    # External dry and wet temperatures for July: hour by hour from 0 to 24h (local solar hour)
    h_sol = np.arange(25).astype(np.float32)

    #Atmospheric pressure at sea level [Pa]
    p_0 = 101325 
    #Estimation of atmospheric pressure at local height
    #Scale height of the Rayleigh atmosphere near the earth surface [m]"
    z_h = 8434.5
    #Local height above the sea level [m]"
    z_local = 100 
    p_atm = exp(-z_local/z_h)*p_0

    np.set_printoptions(edgeitems=25)

    phi_deg = 50.8411        # Latitude
    lambda_deg = -5.5     # Longitude

    n_days_year =  365

    pi = np.pi

    # Longitude expressed in hours
    lambda_h = lambda_deg/15

    sin_phi = sin(phi_deg * pi/180)
    cos_phi = cos(phi_deg * pi/180)
    tan_phi = tan(phi_deg * pi/180)

    hour_sol_local = hour_yr

    # Equation of time ET in hours
    # beta = 2*pi/365 in rad/day, J = hour_sol_local/24, betaJ in rad
    betaJ = (2*pi/n_days_year)*(hour_sol_local/24)
    ET    = (1/60) * (-0.00037+0.43177*np.cos(betaJ) - 3.165*np.cos(2*betaJ) - 0.07272*np.cos(3*betaJ) \
                - 7.3764*np.sin(betaJ) - 9.3893*np.sin(2*betaJ) - 0.24498*np.sin(3*betaJ))

    hour_sol_local_per = hour_sol_local-24*np.trunc(hour_sol_local/24)
    # Assign 24 h  to the zero h elements 
    hour_sol_local_per[hour_sol_local_per == 0] = 24
    # day=1+np.trunc(hour_sol_local/24)

    time_rad=2*pi*hour_sol_local/(24*365)
    cos_time=np.cos(time_rad)

    # hour_south_per = heure périodique égale à 0h quand le soleil est au Sud (azimut gamma = 0)
    hour_south_per = hour_sol_local_per - 12

    # Angle horaire omega en degres : omega = 0 quand le soleil est au Sud (azimut gamma = 0)
    omega_deg = hour_south_per*15   
    sin_omega = np.sin(omega_deg * pi/180)
    cos_omega = np.cos(omega_deg * pi/180)

    # Sun declination delta en degres
    time_rad=2*pi*hour_sol_local/(24*n_days_year)
    time_lag_rad = 2*pi*(284/n_days_year)
    sin_time_decl = np.sin(time_rad+time_lag_rad)
    delta_rad=0.40928*sin_time_decl
    delta_deg=(180/pi)*delta_rad

    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    tan_delta = np.tan(delta_rad)

    # Angle theta_z between sun beams and vertical"
    theta_z_rad = np.abs(np.arccos(sin_delta*sin_phi+cos_delta*cos_phi*cos_omega))
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    theta_z_deg= (180/pi)*theta_z_rad

    # Compute gamma_s : Sun azimuth "
    # Azimut value comprised between -pi and +pi
    gamma_s_rad = np.arctan2(sin_omega, cos_omega * sin_phi - tan_delta * cos_phi)

    sin_gamma_s = np.sin(gamma_s_rad)
    cos_gamma_s = np.cos(gamma_s_rad)
    # Azimut value comprised between -180 and +180
    gamma_s_deg = (180/pi)*gamma_s_rad 

    # Components of the unit vector parallel to sun  beams in axes: South, East, Vertical
    n_sun_beam_South = cos_gamma_s*sin_theta_z
    n_sun_beam_East = sin_gamma_s*sin_theta_z
    n_sun_beam_Vert = cos_theta_z

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.choose(theta_z_deg < 90,[0, 90 - theta_z_deg])
    h_s_rad = (pi/180) * h_s_deg

    # Compute I_th_cs from the solar radiation I_dot_n_0 external to the atmosphere

    # Direct normal irradiance calculation
    # Solar constant [W/m^2]
    I_dot_0 = 1367  
    # Correction for the variation of sun-earth distance [W/m^2]
    delta_I_dot_0 = 45.326 
    I_dot_n_0 = I_dot_0 + delta_I_dot_0*cos_time

    I_th_0= np.where(cos_theta_z > 0,I_dot_n_0*cos_theta_z,0)

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.where(theta_z_deg < 90,90 - theta_z_deg , 0)
    h_s_rad = (pi/180) * h_s_deg

    # Correction of solar altitude angle for refraction
    DELTAh_s_deg = 0.061359*(180/pi)*(0.1594+1.1230*(pi/180)*h_s_deg+0.065656*(pi/180)**2 * h_s_deg**2)/(1+28.9344*(pi/180)*h_s_deg+277.3971*(pi/180)**2 *h_s_deg**2)
    h_s_true_deg = h_s_deg + DELTAh_s_deg
    h_s_true_rad = (pi/180)* h_s_true_deg

    # m_r: relative optical air mass
    # m_r: ratio of the optical path lenght through atmosphere "
    # and the optical path lenght through a standard atmosphere at sea level with the sun at the zenith"
    # Kasten and Young (1989)"
    m_r = p_atm / p_0 /(np.sin(h_s_true_rad) + 0.50572 *((h_s_true_deg + 6.07995)**(-1.6364)))

    # delta_R: Integral Rayleigh optical thickness"
    delta_R = np.where(m_r > 20, 1/(10.4+0.718*m_r), 1/ (6.62960+1.75130*m_r-0.12020*m_r**2+0.00650*m_r**3-0.00013*m_r**4))

    # T_L_2: Linke turbidity factor for relative air mass = 2"
    # Site turbidity beta_site: Country: 0.05  Urban:0.20"
    beta_site = 0.05

    T_L_summer = 3.302
    T_L_winter = 2.455
    T_L_avg=(T_L_summer+T_L_winter)/2
    DELTAT_L=(T_L_summer-T_L_winter)/2
    time_lag_w_deg=360*(-30.5/360-1/4)
    time_lag_w_rad=2*pi*(-30.5/360-1/4)
    sin_time_w = np.sin(time_rad+time_lag_w_rad)
    T_L_2=T_L_avg+DELTAT_L*sin_time_w

    # Direct horizontal irradiance"
    I_bh_cs = I_dot_n_0 * np.sin(h_s_rad) * np.exp(-0.8662*T_L_2*m_r*delta_R)

    # Direct normal irradiance
    # Not considered if sun flicking the wall with an angle < 2°
    I_beam_cs = np.where(cos_theta_z > 0.035, I_bh_cs / cos_theta_z, 0)

    # Diffuse horizontal irradiance"
    # T_rd: diffuse transmission for sun in zenith"
    T_rd = -1.5843E-2+3.0543E-2*T_L_2+3.797E-4*T_L_2**2

    # F_d: diffuse angular function"
    A_0 = 2.6463E-1-6.1581E-2*T_L_2+3.1408E-3*T_L_2**2
    A_1 = 2.0402+1.8945E-2*T_L_2-1.1161E-2*T_L_2**2
    A_2 = -1.3025+3.9231E-2*T_L_2+8.5079E-3*T_L_2**2
    F_d = A_0+A_1*np.sin(h_s_rad)+A_2*(np.sin(h_s_rad)**2)

    # Diffuse horizontal irradiance"
    I_dh_cs = np.where(h_s_deg > 2, I_dot_n_0*T_rd*F_d, 0)
    I_test = I_dot_n_0*T_rd*F_d

    # Total horizontal irradiance"
    I_th_cs= I_bh_cs+I_dh_cs

    # # Calibration from data of Printemps site
    # I_th_cs= 1.1274044452626812 *I_th_cs_0

    I_bh = I_bh_cs
    I_dh = I_dh_cs
    I_th = I_th_cs

    theta_z_rad = theta_z_deg * (pi/180)
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    tan_theta_z= np.tan(theta_z_rad)
    
    return I_bh, I_dh, I_th, theta_z_rad , cos_theta_z, sin_theta_z, tan_theta_z, gamma_s_deg, theta_z_deg, n_sun_beam_South, n_sun_beam_East, n_sun_beam_Vert



def Iwd(azimuth_wd_deg, H_B, A1, A2, A3, D_H, month):

    slope_wd_deg = 90

    angle_lateral_screens_deg = A1
    angle_horiz_screen_deg = A2
    f_prop_dist_vert_screen = D_H
    if f_prop_dist_vert_screen > 0: 
        iflag_vert_screen= 1 
    else:
        iflag_vert_screen= 0 
    angle_vert_screen_deg=min(A3,85)

    H_B_wd=max(0.001,H_B)
    area_wd = Height_wd_0 * Breadth_wd_0
    Height_wd = (area_wd * H_B_wd) ** 0.5
    Breadth_wd = (area_wd / H_B_wd) ** 0.5
    Height_wd_sill = max(0.1, min(Height_wd_sill_0, Height_room - Height_wd) )
    
    # "!Window area"
    area_wd=Height_wd*Breadth_wd
    SF_wd=(1-f_frame)*SF_gl

    # "!Walls areas "
    area_wall_0h=max(0,Height_room*Lenght_9h_3h-area_wd)
    area_wall_3h=Height_room*Lenght_6h_0h
    area_wall_6h=Height_room*Lenght_9h_3h
    area_wall_9h=Height_room*Lenght_6h_0h
    area_ceiling=Lenght_9h_3h*Lenght_6h_0h
    area_floor=Lenght_9h_3h*Lenght_6h_0h

    n_walls=6
    area_wall = np.array([area_wall_0h, area_wall_3h, area_wall_6h, area_wall_9h, area_ceiling, area_floor])
    area_wall_wd=np.sum(area_wall)+area_wd
 
    # Define the gamma_w : wall azimuth
    gamma_w_deg = azimuth_wd_deg
    gamma_w_rad = gamma_w_deg*pi/180
    sin_gamma_w = sin(gamma_w_rad)
    cos_gamma_w = cos(gamma_w_rad)

    # Define the p : slope
    p_deg = slope_wd_deg
    p_rad = p_deg *pi/180
    sin_p = sin(p_rad)
    cos_p = cos(p_rad)
    
    # Ground reflexion: grass 0.2, snow 0.9
    rho_ground=0

    # Difference of azimuth between sun and wall
    # deltagamma_deg comprised between 0° and 180°
    # Where True, yield x, otherwise yield y.
    deltagamma_deg = np.where(abs(gamma_s_deg - gamma_w_deg) > 180, abs(abs(gamma_s_deg - gamma_w_deg) - 360),
                            abs(gamma_s_deg - gamma_w_deg))
    
    I_wd = np.zeros(len(gamma_s_deg))
    I_wd_no_shading = np.zeros(len(gamma_s_deg))
    
    for i in range(len(gamma_s_deg)):
        if ((theta_z_deg[i] < 88) & (I_th[i] > 1) & (area_wd > 0.01)):
            
            # deltagamma_rad comprised between 0 and pi
            deltagamma_rad = deltagamma_deg[i] *pi/180
            cos_dgamma = np.cos(deltagamma_rad)
            sin_dgamma = np.sin(deltagamma_rad)
            tan_dgamma = np.tan(deltagamma_rad)

            # Compute ratio= cos(theta)/cos(theta_z)
            # Cos of angle theta between sun beams and normal direction to the wall"
            # Mask effect if sun flicking the wall with an angle < 2°
            cos_theta = cos_dgamma * sin_theta_z[i]
            cos_theta_cos_theta_z = cos_theta / cos_theta_z[i] 
            I_beam = I_bh[i]/ cos_theta_z[i]  if (I_bh[i] > 0) else 0
            I_b = cos_theta * I_beam if (cos_theta > 0) else 0

            # Diffuse and reflected solar gains"
            I_dr = I_dh[i] * (1+cos_p) / 2 + I_th[i] * rho_ground * (1-cos_p)/2

            # Solar irradiance on the facade"
            I_tv = I_b + I_dr

            I_t_wd = I_tv

            #Diffuse and reflected radiation on vertical plane 
            I_d_wd=I_dh[i]/2

            # Shading factor"

            # Lateral vertical screens supposed symetrical: horizontal angle measured from the center of the window
            tan_A1                    = tan(angle_lateral_screens_deg*pi/180)
            Depth_lateral_screen      = tan_A1*Breadth_wd/2
            b_wd_shade_lateral_screen = Depth_lateral_screen*abs(tan_dgamma) if(cos_dgamma> 0) else 0

            #Horizontal screen upside the window: vertical angle measured from the center of the window#
            tan_A2                    = tan(angle_horiz_screen_deg*pi/180)
            Depth_horiz_screen        = tan_A2*Height_wd/2
            h_wd_shade_horiz_screen   = Depth_horiz_screen/(tan_theta_z[i]*cos_dgamma) if ((tan_theta_z[i]*cos_dgamma > 0.001)& (cos_dgamma>0)) else 0

            #Vertical screen facing the window: vertical angle measured from the center of the window
            Dist_vert_screen          = f_prop_dist_vert_screen*Height_wd
            Hypoth_vert_screen        = Dist_vert_screen/cos_dgamma if(cos_dgamma > 0.001) else 0
            h_vert_screen_no_shade    = Hypoth_vert_screen/tan_theta_z[i] if(tan_theta_z[i] > 0.001) else 0
            tan_A3                    = tan(angle_vert_screen_deg*pi/180)
            h_vert_screen             = Height_wd/2+Dist_vert_screen*tan_A3
            h_vert_screen_shade       = h_vert_screen-h_vert_screen_no_shade if(h_vert_screen > h_vert_screen_no_shade) else 0
            h_wd_shade_vert_screen    = h_vert_screen_shade if(h_vert_screen_no_shade > 0) else 0

            #Shading factor
            h_wd_shade = h_wd_shade_vert_screen+h_wd_shade_horiz_screen
            dh_shade = Height_wd-h_wd_shade if (Height_wd>h_wd_shade) else 0
            b_wd_shade = b_wd_shade_lateral_screen
            db_shade = Breadth_wd-b_wd_shade if (Breadth_wd>b_wd_shade) else 0
            area_no_shaded_wd=dh_shade *db_shade
            f_no_shaded_wd=area_no_shaded_wd/area_wd
            f_shading=(1-f_no_shaded_wd)

            #! Solar gains through windows taking into account the shading due to external screens, without external blinds
            I_wd[i] = (1-f_shading)*I_t_wd+f_shading*I_d_wd
            I_wd_no_shading[i] = I_t_wd
     
    
    return I_wd, I_wd_no_shading


def tout( month):

    t_dry_july = np.array([20.6 , 20.3, 20.2 , 20.0, 20.1 , 20.3, 20.7, 21.3, 21.9 , 22.5 , 23.2 , \
                           23.8 , 26.6 , 29.2, 31.5 , 32.0, 31.5 , 30.3 , 28. , 25.6, 22.7 , 22.1, 21.6 , 21.1 , 20.6 ])
    
    # # Correction month by month - Max daily External dry and wet temperatures 
    dt_dry_m = np.array([-11. , -10. ,  -7.8,  -5.5,  -2.5,  -0.5,   0. ,   0. ,  -2.5,  -4.1,  -8.2, -10.2])
    dt_wet_m = np.array([-5.5, -5. , -3.9, -2.7, -2.3,  0. ,  0. ,  0. , -0.5, -2.3, -3.9, -5. ])

    dt_dry = dt_dry_m[month-1]
    dt_wet = dt_wet_m[month-1]

    # # External dry and wet temperatures for the current month: hour by hour from 0 to 24h (local solar hour)
    t_dry_std = t_dry_july + dt_dry
    t_dry = t_dry_std
    
    return t_dry


def DTE(alpha_wall, M_A, azimuth_w_deg, slope_w_deg, iflag_shading, month, t_in):

    M_per_A_wall=max(10,M_A)

    t_init=26 #[°C]
    
    # Daily average external temperature, in °C
    t_out_avg  = 24

    # Daily variation of external temperature, in °C
    DELTAt_out = 17 

    # "Air properties:"
    v_a=0.8401 # [m^3/kg] "specific volume of humid air per kg of dry air"
    c_p_a=1020 # [J/kg-K] "specific heat capacity of humid air per kg of dry air"
    sigma_boltzman=5.67E-8

    # "!Boundary layers"
    h_r=5 # [W/m^2-K]
    h_c=3 # [W/m^2-K]
    h_in=h_r + h_c
    h_out=17.5 # [W/m^2-K]

    # "!Outside insulation for external wall "
    R_out= 2 # [m^2-K/W]

    # "!Days of simulation"
    n_day_sim=3

    # Wall azimuth angle gamma is comprised between -180° and 180°#
    if azimuth_w_deg > 180 : 
        gamma_w_deg = azimuth_w_deg-360
    else:
        gamma_w_deg = azimuth_w_deg

    # concrete bloc
    rho_concrete_bloc=1200 #[kg/m^3]
    lambda_concrete_bloc=1.273 #[W/m.K]
    c_p_concrete_bloc=840 #[J/kg.K]

    # "!Total number of finite element layers, with two degree two elements by layer"
    n_layers = 2
    nl=n_layers

    # "! internal vertical wall layers"
    thickness_w = M_per_A_wall/rho_concrete_bloc
    thickness_wall = thickness_w/n_layers * np.ones(n_layers) 
    lambda_wall =  lambda_concrete_bloc * np.ones(n_layers)
    rho_wall =  rho_concrete_bloc * np.ones(n_layers)
    c_layer_wall = c_p_concrete_bloc * np.ones(n_layers)

    # Matrixes of vertical wall layers"
    n_elem,R_nobl_wall,L_wall,C_wall = wall_matrix(n_layers,thickness_wall,lambda_wall,rho_wall,c_layer_wall)

    R_value_wall=1/h_in+R_nobl_wall+1/h_out
    U_value_wall=1/R_value_wall

    n_nodes=2*n_elem+1

    # Initial conditions"
    t_a_in_set = t_in + 0.01 #[°C]
    t_a_in_init=t_a_in_set

    # Simulation period
    n_day_sim=3
    hour_start=0
    n_hours_sim=24*n_day_sim
    hour_stop=hour_start + n_hours_sim

    tau_initial=hour_start*3600
    tau_final=hour_stop*3600

    DELTAtau=600 * 2 #[s]

    # Time in s : Create an array of evenly-spaced values
    tau = np.arange(tau_initial,tau_final+1,DELTAtau)

    # Hour and Day from the start of the simulation
    hour       = tau/3600
    hour_per_0 = hour-24*np.trunc(hour/24)
    # np.choose(condition,[action if condition = 0 or false, action if condition = 1 or true])
    # np.choose(array, [action if condition = 0, action if condition = 1 , action if condition = 2 ...)])
    hour_per=np.choose(hour_per_0 > 0.000001,[24,hour_per_0])

    day = hour/24
    day_int_0 = np.trunc(hour/24)+1
    day_int   = day_int_0-1

    # Sarting hour in sun data table according to month of simulation
    month_start=max(1,min(12,month))
    hour_start = np.zeros(13)
    hour_start[1]=1+15*24; hour_start[2]=1+(31+15)*24; hour_start[3]=1+(31+28+15)*24; hour_start[4]=1+(2*31+28+15)*24; hour_start[5]=1+(2*31+28+30+15)*24; hour_start[6]=1+(3*31+28+30+15)*24
    hour_start[7]=1+(3*31+28+2*30+15)*24; hour_start[8]=1+(4*31+28+2*30+15)*24; hour_start[9]=1+(5*31+28+2*30+15)*24; hour_start[10]=1+(5*31+28+3*30+15)*24
    hour_start[11]=1+(6*31+28+3*30+15)*24; hour_start[12]=1+(6*31+28+4*30+15)*24

    # Hour and Day from the start of the year (simulation starts at 15th of the considered month)
    hour_yr = hour + float(hour_start[month])
    day_yr = hour_yr/24

    # External dry and wet temperatures for July: hour by hour from 0 to 24h (local solar hour)
    h_sol = np.arange(25).astype(np.float32)

#     t_dry_july = np.array([21. , 18.5, 16. , 15.5, 15. , 15.5, 16. , 18.5, 21. , 24. , 27. , \
#                            29. , 31. , 31.5, 32. , 31.5, 31. , 30. , 29. , 27.5, 26. , 24.5, 23. , 22. , 21. ])
#     t_wet_july = np.array([16.15,15.24,14.3,14.11,13.92,14.11,14.3,15.24,16.15,17.21,18.22, \
#                            18.88,19.52,19.67,19.83,19.67,19.52,19.2,18.88,18.39,17.89,17.38,16.86,16.51,16.15])

    t_dry_july = np.array([20.6 , 20.3, 20.2 , 20.0, 20.1 , 20.3, 20.7, 21.3, 21.9 , 22.5 , 23.2 , \
                           23.8 , 26.6 , 29.2, 31.5 , 32.0, 31.5 , 30.3 , 28. , 25.6, 22.7 , 22.1, 21.6 , 21.1 , 20.6 ])
    
    # # Correction month by month - Max daily External dry and wet temperatures 
    dt_dry_m = np.array([-11. , -10. ,  -7.8,  -5.5,  -2.5,  -0.5,   0. ,   0. ,  -2.5,  -4.1,  -8.2, -10.2])
    dt_wet_m = np.array([-5.5, -5. , -3.9, -2.7, -2.3,  0. ,  0. ,  0. , -0.5, -2.3, -3.9, -5. ])

    dt_dry = dt_dry_m[month-1]
    dt_wet = dt_wet_m[month-1]

    # # External dry and wet temperatures for the current month: hour by hour from 0 to 24h (local solar hour)
    t_dry_std = t_dry_july + dt_dry
#     t_wet_std = t_wet_july - dt_wet

#     t_dry_avg_std = np.average(t_dry_std)
#     DELTAt_dry_std = np.max(t_dry_std) - np.min(t_dry_std)

#     t_wet_avg_std = np.average(t_wet_std)
#     DELTAt_wet_std = np.max(t_wet_std) - np.min(t_wet_std)

#     # Profile adapted to the data given for t_out_avg and DELTAt_out
#     t_dry = t_out_avg + (t_dry_std-t_dry_avg_std) * (DELTAt_out/DELTAt_dry_std)
#     t_wet = t_out_avg - (t_dry_avg_std - t_wet_avg_std) + (t_wet_std-t_wet_avg_std) * (DELTAt_out/DELTAt_dry_std)

    t_dry = t_dry_std

    df = pd.DataFrame(tau, columns=['tau'])

    df['hour']     = hour
    df['day']      = day
    df['day_int']  = day_int
    df['hour_yr']  = hour_yr
    df['day_yr']   = day_yr
    df['hour_per'] = hour_per

    df1 = pd.DataFrame(h_sol, columns=['hour_per'])
    df1['t_dry'] = t_dry
#     df1['t_wet'] = t_wet

    df = df.merge(df1, how='left')

    # replace NaN values with interpolated values
    df['t_dry'] = df.interpolate(method = 'values')['t_dry']
#     df['t_wet'] = df.interpolate(method = 'values')['t_wet']

    # df

    #Atmospheric pressure at sea level [Pa]
    p_0 = 101325 
    #Estimation of atmospheric pressure at local height
    #Scale height of the Rayleigh atmosphere near the earth surface [m]"
    z_h = 8434.5
    #Local height above the sea level [m]"
    z_local = 62 
    p_atm = exp(-z_local/z_h)*p_0

    # Relative humidity from t_dry and t_wet
    t_dry = df['t_dry'].values
    # t_wet = df['t_wet'].values
    # p_atm_hPa = p_atm/100
    # rh_out_0 = (np.exp(1.8096+(17.2694*t_wet/(237.3+t_wet))) - 7.866*10**-4 * p_atm_hPa * (t_dry-t_wet) * (1+t_wet/610)) \
    #         /(np.exp(1.8096+(17.2694*t_dry/(237.3+t_dry))))

    # # Where True, yield x, otherwise yield y.
    # rh_out = np.where(rh_out_0 >1, 1, rh_out_0)

    # T_dry = t_dry + 273
    # p_v_sat_out = np.exp(77.3450 + 0.0057 * T_dry- 7235 / T_dry)/ T_dry**8.2
    # p_v_out = rh_out * p_v_sat_out

    np.set_printoptions(edgeitems=25)

    phi_deg = 50.90        # Latitude
    lambda_deg = -4.53     # Longitude

    n_days_year =  365

    pi = np.pi

    # Longitude expressed in hours
    lambda_h = lambda_deg/15

    sin_phi = sin(phi_deg * pi/180)
    cos_phi = cos(phi_deg * pi/180)
    tan_phi = tan(phi_deg * pi/180)

    hour_sol_local = hour_yr

    # Equation of time ET in hours
    # beta = 2*pi/365 in rad/day, J = hour_sol_local/24, betaJ in rad
    betaJ = (2*pi/n_days_year)*(hour_sol_local/24)
    ET    = (1/60) * (-0.00037+0.43177*np.cos(betaJ) - 3.165*np.cos(2*betaJ) - 0.07272*np.cos(3*betaJ) \
                - 7.3764*np.sin(betaJ) - 9.3893*np.sin(2*betaJ) - 0.24498*np.sin(3*betaJ))

    hour_sol_local_per = hour_sol_local-24*np.trunc(hour_sol_local/24)
    # Assign 24 h  to the zero h elements 
    hour_sol_local_per[hour_sol_local_per == 0] = 24
    # day=1+np.trunc(hour_sol_local/24)

    time_rad=2*pi*hour_sol_local/(24*365)
    cos_time=np.cos(time_rad)

    # hour_south_per = heure périodique égale à 0h quand le soleil est au Sud (azimut gamma = 0)
    hour_south_per = hour_sol_local_per - 12

    # Angle horaire omega en degres : omega = 0 quand le soleil est au Sud (azimut gamma = 0)
    omega_deg = hour_south_per*15   
    sin_omega = np.sin(omega_deg * pi/180)
    cos_omega = np.cos(omega_deg * pi/180)

    # Sun declination delta en degres
    time_rad=2*pi*hour_sol_local/(24*n_days_year)
    time_lag_rad = 2*pi*(284/n_days_year)
    sin_time_decl = np.sin(time_rad+time_lag_rad)
    delta_rad=0.40928*sin_time_decl
    delta_deg=(180/pi)*delta_rad

    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    tan_delta = np.tan(delta_rad)

    # Angle theta_z between sun beams and vertical"
    theta_z_rad = np.abs(np.arccos(sin_delta*sin_phi+cos_delta*cos_phi*cos_omega))
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    theta_z_deg= (180/pi)*theta_z_rad

    # Compute gamma_s : Sun azimuth "
    # Azimut value comprised between -pi and +pi
    gamma_s_rad = np.arctan2(sin_omega, cos_omega * sin_phi - tan_delta * cos_phi)

    sin_gamma_s = np.sin(gamma_s_rad)
    cos_gamma_s = np.cos(gamma_s_rad)
    # Azimut value comprised between -180 and +180
    gamma_s_deg = (180/pi)*gamma_s_rad 

    # Components of the unit vector parallel to sun  beams in axes: South, East, Vertical
    n_sun_beam_South = cos_gamma_s*sin_theta_z
    n_sun_beam_East = sin_gamma_s*sin_theta_z
    n_sun_beam_Vert = cos_theta_z

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.choose(theta_z_deg < 90,[0, 90 - theta_z_deg])
    h_s_rad = (pi/180) * h_s_deg

    # Compute I_th_cs from the solar radiation I_dot_n_0 external to the atmosphere

    # Direct normal irradiance calculation
    # Solar constant [W/m^2]
    I_dot_0 = 1367  
    # Correction for the variation of sun-earth distance [W/m^2]
    delta_I_dot_0 = 45.326 
    I_dot_n_0 = I_dot_0 + delta_I_dot_0*cos_time

    I_th_0= np.where(cos_theta_z > 0,I_dot_n_0*cos_theta_z,0)

    # Direct horizontal irradiance calculation
    # Solar altitude angle: only when the sun is over the horizontal plane
    h_s_deg = np.where(theta_z_deg < 90,90 - theta_z_deg , 0)
    h_s_rad = (pi/180) * h_s_deg

    # Correction of solar altitude angle for refraction
    DELTAh_s_deg = 0.061359*(180/pi)*(0.1594+1.1230*(pi/180)*h_s_deg+0.065656*(pi/180)**2 * h_s_deg**2)/(1+28.9344*(pi/180)*h_s_deg+277.3971*(pi/180)**2 *h_s_deg**2)
    h_s_true_deg = h_s_deg + DELTAh_s_deg
    h_s_true_rad = (pi/180)* h_s_true_deg

    # m_r: relative optical air mass
    # m_r: ratio of the optical path lenght through atmosphere "
    # and the optical path lenght through a standard atmosphere at sea level with the sun at the zenith"
    # Kasten and Young (1989)"
    m_r = p_atm / p_0 /(np.sin(h_s_true_rad) + 0.50572 *((h_s_true_deg + 6.07995)**(-1.6364)))

    # delta_R: Integral Rayleigh optical thickness"
    delta_R = np.where(m_r > 20, 1/(10.4+0.718*m_r), 1/ (6.62960+1.75130*m_r-0.12020*m_r**2+0.00650*m_r**3-0.00013*m_r**4))

    # T_L_2: Linke turbidity factor for relative air mass = 2"
    # Site turbidity beta_site: Country: 0.05  Urban:0.20"
    beta_site = 0.05

    T_L_summer = 3.302
    T_L_winter = 2.455
    T_L_avg=(T_L_summer+T_L_winter)/2
    DELTAT_L=(T_L_summer-T_L_winter)/2
    time_lag_w_deg=360*(-30.5/360-1/4)
    time_lag_w_rad=2*pi*(-30.5/360-1/4)
    sin_time_w = np.sin(time_rad+time_lag_w_rad)
    T_L_2=T_L_avg+DELTAT_L*sin_time_w

    # Direct horizontal irradiance"
    I_bh_cs = I_dot_n_0 * np.sin(h_s_rad) * np.exp(-0.8662*T_L_2*m_r*delta_R)

    # Direct normal irradiance
    # Not considered if sun flicking the wall with an angle < 2°
    I_beam_cs = np.where(cos_theta_z > 0.035, I_bh_cs / cos_theta_z, 0)

    # Diffuse horizontal irradiance"
    # T_rd: diffuse transmission for sun in zenith"
    T_rd = -1.5843E-2+3.0543E-2*T_L_2+3.797E-4*T_L_2**2

    # F_d: diffuse angular function"
    A_0 = 2.6463E-1-6.1581E-2*T_L_2+3.1408E-3*T_L_2**2
    A_1 = 2.0402+1.8945E-2*T_L_2-1.1161E-2*T_L_2**2
    A_2 = -1.3025+3.9231E-2*T_L_2+8.5079E-3*T_L_2**2
    F_d = A_0+A_1*np.sin(h_s_rad)+A_2*(np.sin(h_s_rad)**2)

    # Diffuse horizontal irradiance"
    I_dh_cs = np.where(h_s_deg > 2, I_dot_n_0*T_rd*F_d, 0)
    I_test = I_dot_n_0*T_rd*F_d

    # Total horizontal irradiance"
    I_th_cs= I_bh_cs+I_dh_cs

    I_bh = I_bh_cs
    I_dh = I_dh_cs
    I_th = I_th_cs

    theta_z_rad = theta_z_deg * (pi/180)
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    tan_theta_z= np.tan(theta_z_rad)

    # Define the gamma_w : wall azimuth
    gamma_w_rad = gamma_w_deg*pi/180
    sin_gamma_w = sin(gamma_w_rad)
    cos_gamma_w = cos(gamma_w_rad)

    # Define the p : slope
    p_deg = slope_w_deg
    p_rad = p_deg *pi/180
    sin_p = sin(p_rad)
    cos_p = cos(p_rad)

    # Components of the wall unit normal vector in axes: South, East, Vertical
    n_wall_South = cos_gamma_w*sin_p
    n_wall_East = sin_gamma_w*sin_p
    n_wall_Vert = cos_p

    # Mask effect
    prod_scal_v_n = n_sun_beam_South * n_wall_South + n_sun_beam_East * n_wall_East + n_sun_beam_Vert * n_wall_Vert
    # Sun beams hitting the wall with an angle >2°
    iflag_no_mask = np.where(prod_scal_v_n > 0.035, 1, 0)

    # Difference of azimuth between sun and wall
    # deltagamma_deg comprised between 0° and 180°
    # Where True, yield x, otherwise yield y.
    deltagamma_deg = np.where(abs(gamma_s_deg - gamma_w_deg) > 180, abs(abs(gamma_s_deg - gamma_w_deg) - 360),
                              abs(gamma_s_deg - gamma_w_deg))
    # deltagamma_rad comprised between 0 and pi
    deltagamma_rad = deltagamma_deg *pi/180
    cos_dgamma = np.cos(deltagamma_rad)
    sin_dgamma = np.sin(deltagamma_rad)
    tan_dgamma = np.tan(deltagamma_rad)

    # Compute ratio= cos(theta)/cos(theta_z)
    # Cos of angle theta between sun beams and normal direction to the wall"
    # Mask effect if sun flicking the wall with an angle < 2°
    cos_theta = cos_dgamma * sin_theta_z
    # Where True, yield x, otherwise yield y.
    cos_theta_cos_theta_z = np.where(sin_theta_z > 0.035, cos_theta / cos_theta_z, 0)

    I_beam = np.where(cos_theta_z > 0.035, I_bh / cos_theta_z, 0)
    I_b = np.where(cos_theta > 0.035, cos_theta * I_beam, 0)

    # Ground reflexion: grass 0.2, snow 0.9"
    rho_ground=0.2

    # Diffuse and reflected solar gains"
    I_dr = I_dh * (1+cos_p) / 2 + I_th * rho_ground * (1-cos_p)/2

    # Solar irradiance on the facade"
    I_tv = I_b + I_dr

    # Ground reflexion
    rho_ground=0

    # Diffuse and reflected solar gains
    I_dr_cs=I_dh_cs*(1+cos_p)/2+I_th_cs*rho_ground*(1-cos_p)/2
    #direct solar gains#
    I_b_cs=iflag_no_mask*I_bh_cs*np.where((cos_p+sin_p*cos_theta_cos_theta_z) > 0, (cos_p+sin_p*cos_theta_cos_theta_z), 0)
    # Solar gains on the wall
    I_t_cs=I_b_cs+I_dr_cs

    I_wall=(1- iflag_shading)*I_t_cs+iflag_shading*I_dr_cs
    t_eq_out_wall= t_dry +alpha_wall*I_wall/h_out

    tau_0  = tau[0]

    Ti0  = t_init * np.ones(n_nodes)

    Q_dot_cooling_v = []

    # t_a_in_set = 26.01


    def model_dTi_t(Ti, tau):
        Tw = Ti[0:]

        ind = int((tau - tau_0) /DELTAtau) 
        if ind>len(t_eq_out_wall)-1: ind=len(t_eq_out_wall)-1

        # Radiative and convective heat exchanges at wall surfaces
        t_s_wall_in = Tw[0]
        t_s_wall_out = Tw[-1]

        #! All walls
        shape = (n_nodes)
        fh = np.zeros(shape)
        dTw_t = np.zeros(shape)

        # Inside boundary conditions
        fh[0] = h_in*(t_a_in_set-t_s_wall_in)

        # Outside boundary conditions
        fh[-1] = h_out*(t_eq_out_wall[ind]-t_s_wall_out)

        dTw_t[:] = np.linalg.inv(C_wall[:,:]) @ (fh[:] - L_wall[:,:] @ Tw[:])

        dTi_t = dTw_t

        q_dot_in_to_wall_A = fh[0]
        q_dot_wall_to_in_A = - fh[0]
        DELTAt_eq = q_dot_wall_to_in_A/U_value_wall

        return (dTi_t, DELTAt_eq) # allows to return more outputs than only dTi_T

    def dTi_t(Ti, tau):
        ret = model_dTi_t(Ti, tau) # only dTi_T is returned for ordinary differential integral
        return ret[0]

    DELTAt_ext_int = t_dry-t_a_in_set
    DELTAt_eq_out=t_eq_out_wall-t_a_in_set

    Ti = odeint(dTi_t , Ti0, tau)

    DELTAt_eq = np.asarray([model_dTi_t(Ti[tt],tau[tt])[1] for tt in range(len(tau))])

    DELTAt_eq_max = np.max(DELTAt_eq)
    
    dfl = pd.DataFrame( {'DTE':DELTAt_eq, 'DTE_out':DELTAt_eq_out, 'TOUT':t_dry})
    dfT = pd.DataFrame(Ti[0:],columns=['T_' + str(j+1) for j in range(n_nodes)])
    dfl = pd.merge(dfl, dfT, left_index=True, right_index=True)
    
    dfl['tau'] = tau
    dfl['hour']= dfl['tau']/3600
    dfl = dfl.drop(['tau'], axis=1)
    dfl['hour_per']= hour_per

    dfl.index = dfl['hour']
    dfl = dfl.loc[n_hours_sim-24:n_hours_sim]

    dfl['hour_per'][0] = 0
    
    dfp = dfl.copy()
    dfp['inthour']= dfp['hour'].astype(int)
    dfp = dfp.loc[dfp['hour'] == dfp['inthour']]
    dfp['hour_per'] = dfp['hour_per'].astype(int)
    dfp.index = dfp['hour_per']
    dfp = dfp[['DTE']].copy()
    dfp['DTE'] = round(dfp['DTE'],2)

    dfc = dfp.copy()
    dfc.index = dfc['DTE']
    dfc = dfc.drop(['DTE'], axis=1)
    
    dfpa = dfl.copy()
    dfpa['inthour']= dfpa['hour'].astype(int)
    dfpa = dfpa.loc[dfpa['hour'] == dfpa['inthour']]
    dfpa['hour_per'] = dfpa['hour_per'].astype(int)
    dfpa.index = dfpa['hour_per']
    dfpa = dfpa[['TOUT']].copy()
    dfpa['TOUT'] = round(dfpa['TOUT'],2)

    dfca = dfpa.copy()
    dfca.index = dfca['TOUT']
    dfca = dfca.drop(['TOUT'], axis=1)
    
#     return dfl, dfp, dfc, dfpa, dfca
    return DELTAt_eq



# Infinitely long, directly opposed parallel plates of the same finite width.

def VF_para_inf(h,w):
    # h : Distance between plates
    # w : Width of the plates
    if w == 0:
        VF = 0
    else:
        HW = h/w
        VF = sqrt(1 + HW**2 ) - HW
    return VF

# Infinitely long, perpendicular plates, from plate 1 to plate 2.

def VF_per_inf(a1,a2):
    # a1 : Width of plate 1
    # a2 : Width of plate 2
    if a1 == 0:
        VF = 0
    else:
        HW = a2/a1
        VF = 1/2*(1 + HW - sqrt (1 + HW**2 ))
    return VF

# View factor parallel plates from 1 to 2 
#(Ref. Isidoro Martinez,"Radiative view Factors", p. 23 )


def BB(x,y,eta,xi,z):
    BB = ((y-eta)*sqrt((x-xi)**2+z**2)*atan((y-eta)/sqrt((x-xi)**2+z**2)) \
          +(x-xi)*sqrt((y-eta)**2+z**2)*atan((x-xi)/sqrt((y-eta)**2+z**2))-z**2/2*log((x-xi)**2+(y-eta)**2+z**2))
    return BB

# a1, b1, a2, b2 : half breadths (along x) and half lenghts (along y) of surfaces 1 and 2
# a0, b0 : coordinates of the center of surface 1 projected on surface 2, refering to surface 2 center
def VF_para(a1, b1, a2, b2, a0, b0, d):
        if d != 0 :
            z = d
            x = np.array([a0 - a1, a0 + a1])
            y = np.array([b0 - b1, b0 + b1])
            xi = np.array([- a2, a2])
            eta = np.array([- b2, b2])
            F = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            F = F + ((-1)**(4+i+j+k+l)) * BB(x[i],y[j],eta[k],xi[l],z)
            A=(x[1]-x[0])*(y[1]-y[0])
            VF = F / (2 * pi * A)

        else:
            VF =0
        return VF

# View factor finite parallel plates same size
def VF_para_id(a,b,c):
    X = a/c
    Y = b/c
    VF = VF_para(a/2, b/2, a/2, b/2, 0, 0, c)
    return VF

# View factor perpendicular plates from 1 to 2 without offsets
def VF_per(a,b,c):
    if a != 0 and b != 0 and c != 0 :
        H=b/c
        W=a/c
        A=((1+W**2)*(1+H**2))/(1+W**2+H**2)
        B=((W**2*(1+W**2+H**2))/((1+W**2)*(W**2+H**2)))**(W**2)
        C=((H**2*(1+W**2+H**2))/((1+H**2)*(W**2+H**2)))**(H**2)
        VF=1/(pi*W)*(W*atan(1/W)+H*atan(1/H)-sqrt(H**2+W**2)*atan(1/sqrt(H**2+W**2))+ (1/4)* log(A*B*C))
    else:
        VF =0
    return VF

# View factor perpendicular plates from 1 to 2 with offsets
def VF_perp(a_1,a_2,b_1,b_2,c):
    A_1 = a_2*c
    A_13= (a_1+a_2)*c
    A_3 = a_1*c
    A_2 = b_2*c
    A_24= (b_1+b_2)*c
    A_4 = b_1*c
    if a_1 == 0 and  b_1 == 0:
        VF = VF_per(a_2,b_2,c)
    elif a_1 == 0:
        VF = VF_per(a_2,b_1+b_2,c) - VF_per(a_2,b_1,c)
    elif b_1 == 0:
        VF = (A_13*VF_per(a_1+a_2,b_2,c)-A_3*VF_per(a_1,b_2,c))/A_1
    else:
        VF=(A_13*VF_per(a_1+a_2,b_1+b_2,c)+A_3*VF_per(a_1,b_1,c) \
            -A_3*VF_per(a_1,b_1+b_2,c)-A_13*VF_per(a_1+a_2,b_1,c))/A_1
    return VF

# View factor between two not touching perpendicular rectangles from 1 to 2
def VF_perp_nt(a1, a2, b1, b2, c1, c2, c3):
# a1, a2, b1, b2, c1, c2, and  c3:  dimensions as shown in the figure.
    BB012345=VF_per((a1+a2),(b1+b2),(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    BB1234=VF_per((a1+a2),(b1+b2),(c1+c2))*(a1+a2)*(c1+c2)
    BB0145=VF_per((a1+a2),(b1+b2),(c2+c3))*(a1+a2)*(c2+c3)
    BB345=VF_per(a1,b1,(c1+c2+c3))*a1*(c1+c2+c3)
    B345_012345=VF_per(a1,(b1+b2),(c1+c2+c3))*a1*(c1+c2+c3)
    B012345_345=VF_per((a1+a2),b1,(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    B0145_45=VF_per((a1+a2),b1,(c2+c3))*(a1+a2)*(c2+c3)
    B1234_34=VF_per((a1+a2),b1,(c1+c2))*(a1+a2)*(c1+c2)
    B45_0145=VF_per(a1,(b1+b2),(c2+c3))*a1*(c2+c3)
    B34_1234=VF_per(a1,(b1+b2),(c1+c2))*a1*(c1+c2)
    BB14=VF_per((a1+a2),(b1+b2),c2)*(a1+a2)*c2
    B14_4=VF_per((a1+a2),b1,c2)*(a1+a2)*c2
    BB45=VF_per(a1,b1,(c2+c3))*a1*(c2+c3)
    BB34=VF_per(a1,b1,(c1+c2))*a1*(c1+c2)
    B4_14=VF_per(a1,(b1+b2),c2)*a1*c2
    BB4=VF_per(a1,b1,c2)*a1*c2
    VF=(BB012345-BB1234-BB0145+BB345-B345_012345-B012345_345+B0145_45+B1234_34+B45_0145+B34_1234+BB14-B14_4 \
        -BB45-BB34-B4_14+BB4)/(2*a2*c3)
    return VF

def View_factors(Lenght_6h_0h,Lenght_9h_3h,Height_room,Breadth_wd,Height_wd_sill,Height_wd):
    
    area_wd = Breadth_wd*Height_wd
    area_wall_0h = Lenght_9h_3h * Height_room - area_wd
    area_wall_3h = Lenght_6h_0h * Height_room
    area_wall_6h = Lenght_9h_3h * Height_room
    area_wall_9h = Lenght_6h_0h * Height_room
    area_ceiling = Lenght_6h_0h * Lenght_9h_3h
    area_floor = Lenght_6h_0h * Lenght_9h_3h
 
    # "! ComputeView factor from window to walls"
    x_1_wd=max(0.001,(Lenght_9h_3h-Breadth_wd)/2)
    x_2_wd=x_1_wd+Breadth_wd
    y_1_wd=Height_wd_sill
    y_2_wd=y_1_wd+Height_wd
 
    FV_wd_to_0h=0
    FV_0h_to_wd=0
 
#     FV_wd_to_6h=VF_para(Breadth_wd/2,Height_wd/2,Lenght_9h_3h/2,Height_room/2,0,0,Lenght_6h_0h)
#     FV_6h_to_wd=VF_para(Lenght_9h_3h/2,Height_room/2,Breadth_wd/2,Height_wd/2,0,0,Lenght_6h_0h)

    a = Height_wd/2 + Height_wd_sill - Height_room/2
    FV_wd_to_6h=VF_para(Breadth_wd/2,Height_wd/2,Lenght_9h_3h/2,Height_room/2,0,-a,Lenght_6h_0h)
    FV_6h_to_wd=VF_para(Lenght_9h_3h/2,Height_room/2,Breadth_wd/2,Height_wd/2,0,a,Lenght_6h_0h)
 
    FV_9h_to_wd_1=VF_perp(0,Lenght_6h_0h,x_1_wd,Breadth_wd,Height_wd)
    FV_9h_to_wd_2=VF_perp_nt(0.001,Lenght_6h_0h,x_1_wd,Breadth_wd,Height_wd,0.0001,y_1_wd)
    FV_9h_to_wd_3=VF_perp_nt(0.001,Lenght_6h_0h,x_1_wd,Breadth_wd,Height_wd,0.0001,Height_room-y_2_wd)
    FV_9h_to_wd=(Height_wd*FV_9h_to_wd_1+y_1_wd*FV_9h_to_wd_2+(Height_room-y_2_wd)*FV_9h_to_wd_3)/Height_room
 
    FV_wd_to_9h=FV_9h_to_wd*area_wall_9h/area_wd
 
    FV_3h_to_wd=FV_9h_to_wd
    FV_wd_to_3h=FV_wd_to_9h
 
    FV_ceiling_to_wd_1=VF_perp(0,Lenght_6h_0h,Height_room-y_2_wd,Height_wd,Breadth_wd)
    FV_ceiling_to_wd_2=VF_perp_nt(0.001,Lenght_6h_0h,Height_room-y_2_wd,Height_wd,Breadth_wd,0.0001,x_1_wd)
    FV_ceiling_to_wd=(Breadth_wd*FV_ceiling_to_wd_1+2*x_1_wd*FV_ceiling_to_wd_2)/Lenght_9h_3h
 
    FV_wd_to_ceiling=FV_ceiling_to_wd*area_ceiling/area_wd
 
    FV_floor_to_wd_1=VF_perp(0,Lenght_6h_0h,y_1_wd,Height_wd,Breadth_wd)
    FV_floor_to_wd_2=VF_perp_nt(0.001,Lenght_6h_0h,y_1_wd,Height_wd,Breadth_wd,0.0001,x_1_wd)
    FV_floor_to_wd=(Breadth_wd*FV_floor_to_wd_1+2*x_1_wd*FV_floor_to_wd_2)/Lenght_9h_3h
 
    FV_wd_to_floor=FV_floor_to_wd*area_floor/area_wd
 
    # "!Compute view factors from wall 1 to wall 2"
    # "Two opposite identical planes"
    FV_floor_to_ceiling=VF_para_id(Lenght_9h_3h,Lenght_6h_0h,Height_room)
    FV_ceiling_to_floor=FV_floor_to_ceiling
 
    # "Floor to vertical walls"
    FV_floor_to_3h=VF_per(Lenght_9h_3h,Height_room,Lenght_6h_0h)
    FV_3h_to_floor=VF_per(Height_room,Lenght_9h_3h,Lenght_6h_0h)
 
    FV_floor_to_6h=VF_per(Lenght_6h_0h,Height_room,Lenght_9h_3h)
    FV_6h_to_floor=VF_per(Height_room,Lenght_6h_0h,Lenght_9h_3h)

    FV_floor_to_9h=FV_floor_to_3h
    FV_9h_to_floor=FV_3h_to_floor
 
    # "Ceiling and floor: identical FV for walls"
    FV_ceiling_to_3h=FV_floor_to_3h
    FV_3h_to_ceiling=FV_3h_to_floor

    FV_ceiling_to_6h=FV_floor_to_6h
    FV_6h_to_ceiling=FV_6h_to_floor

    FV_ceiling_to_9h=FV_ceiling_to_3h
    FV_9h_to_ceiling=FV_3h_to_ceiling
 
    # "Wall to wall: facing walls"
    FV_3h_to_9h=VF_para_id(Lenght_6h_0h,Height_room,Lenght_9h_3h)
    FV_9h_to_3h=FV_3h_to_9h
 
    # "Wall to wall: corners"
    FV_6h_to_9h=VF_per(Lenght_9h_3h,Lenght_6h_0h,Height_room)
    FV_9h_to_6h=VF_per(Lenght_6h_0h,Lenght_9h_3h,Height_room)

    FV_6h_to_3h=FV_6h_to_9h
    FV_3h_to_6h=FV_9h_to_6h
 
    # "!Compute view factors of the wall 0h "
    FV_3h_to_0h=1-(FV_3h_to_wd+FV_3h_to_6h+FV_3h_to_9h+FV_3h_to_ceiling+FV_3h_to_floor)
    FV_6h_to_0h=1-(FV_6h_to_wd+FV_6h_to_3h+FV_6h_to_9h+FV_6h_to_ceiling+FV_6h_to_floor)
    FV_9h_to_0h=1-(FV_9h_to_wd+FV_9h_to_3h+FV_9h_to_6h+FV_9h_to_ceiling+FV_9h_to_floor)
    FV_ceiling_to_0h=1-(FV_ceiling_to_wd+FV_ceiling_to_3h+FV_ceiling_to_6h+FV_ceiling_to_9h+FV_ceiling_to_floor)
    FV_floor_to_0h=1-(FV_floor_to_wd+FV_floor_to_3h+FV_floor_to_6h+FV_floor_to_9h+FV_floor_to_ceiling)

    FV_0h_to_3h=FV_3h_to_0h*area_wall_3h/area_wall_0h
    FV_0h_to_6h=FV_6h_to_0h*area_wall_6h/area_wall_0h
    FV_0h_to_9h=FV_9h_to_0h*area_wall_9h/area_wall_0h
    FV_0h_to_ceiling=FV_ceiling_to_0h*area_ceiling/area_wall_0h
    FV_0h_to_floor=FV_floor_to_0h*area_floor/area_wall_0h
 
    # "!FV_wall(i,j) : View factor from the wall i to the wall j"
    FV_wall = np.array([[       0, FV_0h_to_3h, FV_0h_to_6h, FV_0h_to_9h, FV_0h_to_ceiling, FV_0h_to_floor], \
                      [FV_3h_to_0h,        0, FV_3h_to_6h, FV_3h_to_9h, FV_3h_to_ceiling, FV_3h_to_floor], \
                      [FV_6h_to_0h, FV_6h_to_3h,        0, FV_6h_to_9h, FV_6h_to_ceiling, FV_6h_to_floor], \
                      [FV_9h_to_0h, FV_9h_to_3h, FV_9h_to_6h,        0, FV_9h_to_ceiling, FV_9h_to_floor], \
                      [FV_ceiling_to_0h, FV_ceiling_to_3h, FV_ceiling_to_6h, FV_ceiling_to_9h,   0, FV_ceiling_to_floor], \
                      [  FV_floor_to_0h,   FV_floor_to_3h,   FV_floor_to_6h,   FV_floor_to_9h, FV_floor_to_ceiling, 0]])

    FV_to_wd = np.array([FV_0h_to_wd, FV_3h_to_wd, FV_6h_to_wd, FV_9h_to_wd, FV_ceiling_to_wd, FV_floor_to_wd])
    FV_wd_to = np.array([FV_wd_to_0h, FV_wd_to_3h, FV_wd_to_6h,  FV_wd_to_9h, FV_wd_to_ceiling, FV_wd_to_floor])

    return FV_wall,FV_to_wd,FV_wd_to


def wall_matrix(n_layers,thickness,lambdaval,rho,c_layer):
    
    #! n_div finite elements of order two by layer"
    n_div=2 # mandatory"
    n_elem=n_div*n_layers

    #! n_div finite elements of order two by layer#
    Res=thickness/lambdaval
    R_nobl=np.sum(Res)
    
    # Matrix lines and rows numbered from 1 to 2*n_elem+1 > at the end of the procedure suppress first row and first column

    #!Define coefficients of C (thermal mass) and L (conductivity) symetric matrixes #
    C_el = np.zeros((4,4))
    L_el = np.zeros((4,4))
    C_el[1,1]=2/15 ; C_el[2,2]=8/15 ; C_el[3,3]=2/15 ; C_el[2,1]=1/15 ;C_el[3,1]=-1/30 ; C_el[3,2]=1/15 
    L_el[1,1]=7/3 ; L_el[2,2]=16/3 ; L_el[3,3]=7/3 ; L_el[2,1]=-8/3 ;L_el[3,1]=1/3 ; L_el[3,2]=-8/3

    #! Define H matrix#
    #! Define vectors : RLE=lambda/epaisseur d'élément, RCE=rho*c_p*epaisseur d'élément#
    RLE0=lambdaval/(thickness/n_div)
    RCE0=rho*c_layer*thickness/n_div
    
    RLE= [0]
    RLE.extend(RLE0)
    RCE= [0]
    RCE.extend(RCE0)

    #!Define symetric matrixes M_1=L*deltaT/2+C and M_0=-L*deltaT/2+C   :#
    shape = (2*n_elem+2,2*n_elem+2)
    L = np.zeros(shape)
    C = np.zeros(shape)
    
    #Termes diagonaux extrêmes#
    L[1,1]= RLE[1]*L_el[1,1]
    C[1,1]= RCE[1]*C_el[1,1]

    L[2*n_elem+1,2*n_elem+1]=RLE[n_layers]*L_el[3,3]
    C[2*n_elem+1,2*n_elem+1]=RCE[n_layers]*C_el[3,3]

    #Termes diagonaux aux endroits de superpositions entre couches différentes#
    for k in range(1,n_layers):
        L[2*n_div*k+1,2*n_div*k+1]=RLE[k]*L_el[3,3]+RLE[k+1]*L_el[1,1]
        C[2*n_div*k+1,2*n_div*k+1]=RCE[k]*C_el[3,3]+RCE[k+1]*C_el[1,1]
   
    #Termes diagonaux aux endroits de superpositions entre mêmes matériaux
    #cette boucle ne fonctionne que pour n_div=2, sinon ajouter des lignes}
    for k in range(1,n_layers+1):
        L[2*n_div*k-1,2*n_div*k-1]=2*RLE[k]*L_el[3,3]
        C[2*n_div*k-1,2*n_div*k-1]=2*RCE[k]*C_el[3,3]

    #Sous-matrices des termes [2;1] [3;1] [2;2] [3;2]#
    for k in range(1,n_layers+1):
        #Première sous-couche i=1, seconde i=2#
        for i in range(1,n_div+1):
            #Terme diagonal#
            L[4*k+2*i-6+2,4*k+2*i-6+2]=RLE[k]*L_el[2,2]
            C[4*k+2*i-6+2,4*k+2*i-6+2]=RCE[k]*C_el[2,2]
            #Trois termes sous la diagonale#
            L[4*k+2*i-6+2,4*k+2*i-6+1]=RLE[k]*L_el[2,1]
            C[4*k+2*i-6+2,4*k+2*i-6+1]=RCE[k]*C_el[2,1]
            L[4*k+2*i-6+3,4*k+2*i-6+1]=RLE[k]*L_el[3,1]
            C[4*k+2*i-6+3,4*k+2*i-6+1]=RCE[k]*C_el[3,1]
            L[4*k+2*i-6+3,4*k+2*i-6+2]=RLE[k]*L_el[3,2]
            C[4*k+2*i-6+3,4*k+2*i-6+2]=RCE[k]*C_el[3,2]

    #Symetry#
    for i in range(1,n_elem*2+2):
        for j in range(1,i):
            L[j,i]=L[i,j]
            C[j,i]=C[i,j]
            
    L=L[1:,1:]
    C=C[1:,1:]

    return n_elem,R_nobl,L,C

def DELTAhour_start(hour_0,hour_start_0,hour_stop_0):
 
    if (hour_0==24):
        hour=0
    else:
        hour=hour_0
        
    if (hour_start_0==24):
        hour_start=0
    else:
        hour_start=hour_start_0
        
    if (hour_stop_0==24):
        hour_stop=0
    else:
        hour_stop=hour_stop_0

    #Checks whether an hour is in between a starting and a stopping hour ON THE CLOCK#
    # If so returns the number of hours from the starting hour#
    if (hour_start<hour_stop):
        if (hour>=hour_start) and (hour<=hour_stop):
            DELTAhour_start=hour-hour_start
        else:
            DELTAhour_start=0
    else:
        if ((hour>=hour_start) or (hour<=hour_stop)):
            if (hour>=hour_start):
                DELTAhour_start=hour-hour_start
            else:
                DELTAhour_start=24-hour_start+hour
        else:
            DELTAhour_start=0

    return DELTAhour_start


def plant(hour_per,hour_start,hour_stop,hour_start_occupancy):

    #DELTA_on=number of system working hours per day#
    if (hour_start<=hour_stop):
        DELTA_on=hour_stop-hour_start
    else:
        DELTA_on=24-hour_start+hour_stop

    if (DELTA_on<24):
        #! System intermittent working#
        #DELTAhour=number of hours from the start to the stop of the system#
        #DELTAhour > 0 means that hour_per is comprised between hour_start and hour_stop, ON THE CLOCK !#
        DELTAhour=DELTAhour_start(hour_per,hour_start,hour_stop)
        #DELTAhour_smooth=number of hours of smooth restart from the start of the system to the start of occupancy#
        #DELTAhour_smooth > 0 means that hour_per is comprised between hour_start and hour_start_occupancy, ON THE CLOCK !#
        DELTAhour_smooth=DELTAhour_start(hour_per,hour_start,hour_start_occupancy)
        #DELTAhour_restart=number of hours available to restart the system before occupancy#
        #DELTAhour_restart > 0 means that hour_start_occupancy is comprised between hour_start and hour_stop, ON THE CLOCK !#
        DELTAhour_restart=DELTAhour_start(hour_start_occupancy,hour_start,hour_stop)
        
        if (DELTAhour_restart >0) and (DELTAhour_smooth>0):
            # Smooth starting #
            plant = DELTAhour_smooth/DELTAhour_restart
        else:
            #On/off control#
#             if (DELTAhour > 0) or (hour_per==hour_start) or  ((hour_per==24) and (hour_start==0)):
            if (DELTAhour > 0) : 
                plant=1
            elif (DELTAhour_restart==0) and ((hour_per==hour_start) or  ((hour_per==24) and (hour_start==0))):
                plant=1
            else:
                plant=0

    else:
        #! System continous working#
        plant=1
 
    return plant


def occupancy(hour_0,hour_start_0,hour_stop_0):
 
    if (hour_0==24):
        hour=0
    else:
        hour=hour_0
    if (hour_start_0==24):
        hour_start=0
    else:
        hour_start=hour_start_0
    if (hour_stop_0==24):
        hour_stop=0
    else:
        hour_stop=hour_stop_0

    if (hour_start<hour_stop):
        if (hour>=hour_start) and (hour<=hour_stop):
            occupancy=1
        else:
            occupancy=0
    else:
        if ((hour>=hour_start) or (hour<=hour_stop)):
            occupancy=1
        else:
            occupancy=0

    return occupancy

def azimuth_norm(az):
    # azimuth comprised between -360° and 360°
    az = az - 360 * np.trunc(az/360)
    # azimuth comprised between -180° and 180°"
    if (abs(az) > 180) :
        if (az > 180) : 
            az = az - 360
        else:
            az = az + 360
    return az



I_bh, I_dh, I_th, theta_z_rad , cos_theta_z, sin_theta_z, tan_theta_z, gamma_s_deg, theta_z_deg, \
n_sun_beam_South, n_sun_beam_East, n_sun_beam_Vert = Isol(month)



