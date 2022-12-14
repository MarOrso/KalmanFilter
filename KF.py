#LIBRERIE

import numpy as np 
import sympy as sym
from filterpy.kalman import KalmanFilter 
import scipy
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import matplotlib.pyplot as plt
from matplotlib import style
import pymap3d as pym
import pandas as pd
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.twobody.propagation import cowell as cowell
from poliastro.core.perturbations import J3_perturbation, J2_perturbation
from poliastro.core.propagation import func_twobody
import math
from sklearn import linear_model, datasets
import matplotlib.style as style
from numpy.linalg import inv, det
from utility.utils import *
from utility.MieFunzioni import *
from ahrs.filters import AngularRate

###########################

dt = 10
mi = 4.9048695e3 # km^3/s^2 
S = 0.006 # m (focal length 700 mm per la wide angle camera)
FOV=61.4 #° WIDE ANGLE CAMERA

#############################
#CONDIZIONI INIZIALI

#Lat, Long, Alt
df = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\LLA.csv") 
real_Latitudes, real_Longitudes, real_Altitudes = df['Lat (deg)'], df['Lon (deg)'], df['Alt (km)']
time = df['Time (UTCG)']

real_X, real_Y, real_Z = [], [], []
for i in range(len(df)):
    altitude = real_Altitudes[i]
    latitude = real_Latitudes[i]
    longitude = real_Longitudes[i]
    x, y, z = spherical2cartesian(altitude, latitude, longitude)
    real_X.append(x)
    real_Y.append(y)
    real_Z.append(z)
real_X, real_Y, real_Z = np.array(real_X),np.array(real_Y),np.array(real_Z)

#Quaternione e velocita' angolare 
dq = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2 - Copia\orbite\Orbit1_nadir\Orbit1_nadir\Attitude_Quaternions_Fixed.csv")  
real_q1, real_q2, real_q3, real_q4 = dq['q1'], dq['q2'], dq['q3'], dq['q4']
real_om1, real_om2, real_om3 = dq['wx (deg/sec)'], dq['wy (deg/sec)'], dq['wz (deg/sec)']

#Velocita'
dvf = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\Velocity_Fixed.csv") 
real_Vxs,real_Vys,real_Vzs = dvf['vx (km/sec)'], dvf['vy (km/sec)'],dvf['vz (km/sec)']

# Initial State Vector:
init_x, init_y, init_z = real_X[0], real_Y[0], real_Z[0]
init_vx, init_vy, init_vz = real_Vxs[0], real_Vys[0], real_Vzs[0]
init_q1, init_q2, init_q3, init_q4 = real_q1[0], real_q2[0], real_q3[0], real_q4[0]
init_om1, init_om2, init_om3 = real_om1[0], real_om2[0], real_om3[0]

#Creo il filtro di Kalman

#State transition matrix
F=find_F_matrix(init_x, init_y, init_z, init_q1, init_q2, init_q3, init_q4, init_om1, init_om2, init_om3) 
phii= np.eye(12)+F*dt 

#Process noise matrix
q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001)
Q = block_diag(q, q, q, q)   
 
X = np.array([[init_x, init_y, init_z, init_vx, init_vy, init_vz, init_q1, init_q2, init_q3,init_om1, init_om2, init_om3]]).T

P=np.diag([3.65996971, 0.29041806, 682008.0961273, 5.52247218, -18.90088871, 0.23225206, 2.8019472, 2.73023895, 3.0819458, 3.35880549, 1.62408328, 1.5544467])


####################
#CREAZIONE DELLE MISURE

#POSIZIONI DELL'ASSE DI NADIR DEL SATELLITE SUL SUOLO IN TERMINI ENU
ENU_E, ENU_N, ENU_U=[],[],[]
real_X_suolo, real_Y_suolo, real_Z_suolo = [], [], []
for i in range(len(df)): #voglio trovare le posizioni del satellite al suolo in ogni istante considerato, cioè 
                         #lungo tutta l'orbita, ogni 10 secondi
    altitude = 0
    latitude = real_Latitudes[i]
    longitude = real_Longitudes[i]
    x, y, z = spherical2cartesian(altitude, latitude, longitude)
    real_X_suolo.append(x)
    real_Y_suolo.append(y)
    real_Z_suolo.append(z)
real_X_suolo, real_Y_suolo, real_Z_suolo = np.array(real_X_suolo),np.array(real_Y_suolo),np.array(real_Z_suolo)

for i in range(len(df)): #trasformo nel riferimento ENU
    latitude = real_Latitudes[i]
    longitude = real_Longitudes[i]
    x= real_X_suolo[i]
    y= real_Y_suolo[i]
    z= real_Z_suolo[i]
    E, N, U = LCLF2ENU (x, y, z, latitude, longitude)
    ENU_E.append(E)
    ENU_N.append(N)
    ENU_U.append(U)
ENU_E, ENU_N, ENU_U = np.array(ENU_E),np.array(ENU_N),np.array(ENU_U)

#LIMITI IN ENU
ES, ED, NS, NG = [], [], [], []
for i in range(len(df)): #Sto ora calcolando ogni 10 secondi i limiti dell'area rettangolare dell'immagine presa 
                         #dalla fotocamera, in termini ENU
    H = real_Altitudes[i]
    E = ENU_E[i]
    N = ENU_N[i]
    d = SW_nadir(H)
    A=E-0.5*d
    B=E+0.5*d
    C=N-0.5*d
    D=N+0.5*d
    ES.append(A) 
    ED.append(B) 
    NS.append(D) 
    NG.append(C) 
ES, ED, NS, NG = np.array(ES),np.array(ED),np.array(NS), np.array(NG)

#LAT E LONG PUNTI RETTANGOLO
Lat_A, Long_A, Lat_B, Long_B = [], [], [], []
Lat_C, Long_C, Lat_D, Long_D = [], [], [], []
for i in range(len(df)):        #Trasformo le coordinate ENU degli estremi del rettangolo in lat, long, alt
    latitude = real_Latitudes[i]
    longitude = real_Longitudes[i]
    U=ENU_U[i]
    #Punto A
    E_A = ES[i]
    N_A = NS[i]
    #Punto B
    E_B = ED[i]
    N_B = NS[i]
    #Punto C
    E_C = ES[i]
    N_C = NG[i]
    #Punto D
    E_D = ED[i]
    N_D = NG[i]
                                         #(est,nord,up,lat,long,alt)
    lat_a, long_a, alt_a=pym.enu2geodetic(E_A,N_A,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))

    Lat_A.append(lat_a)
    Long_A.append(long_a)

    lat_b, long_b, alt_b=pym.enu2geodetic(E_B,N_B,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))
    
    Lat_B.append(lat_b)
    Long_B.append(long_b)

    lat_c, long_c, alt_c=pym.enu2geodetic(E_C,N_C,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))
    
    Lat_C.append(lat_c)
    Long_C.append(long_c)

    lat_d, long_d, alt_d=pym.enu2geodetic(E_D,N_D,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))
    
    Lat_D.append(lat_d)
    Long_D.append(long_d)    
Lat_A, Long_A, Lat_B, Long_B = np.array(Lat_A),np.array(Long_A),np.array(Lat_B), np.array(Long_B)  
Lat_C, Long_C, Lat_D, Long_D = np.array(Lat_C),np.array(Long_C),np.array(Lat_D), np.array(Long_D)  
#Lat_A=Lat_B Lat_C=Lat_D
#Long_A=Long_C Long_B=Long_D

lat_inf, lat_sup, long_inf, long_sup = [], [], [], []
for i in range(len(df)): #Cerco i range di latitudine e longitudine dell'immagine in cui cercare i crateri ogni 10 secondi
    phi_A=Lat_A[i]
    phi_C=Lat_C[i]

    lambd_A=Long_A[i]
    lambd_B=Long_B[i]

    if phi_A<phi_C:
        lat_inf1=phi_A
        lat_sup1=phi_C
    else:         
        lat_inf1=phi_C
        lat_sup1=phi_A

    if lambd_A<lambd_B:
        long_inf1=lambd_A
        long_sup1=lambd_B
    else:         
        long_inf1=lambd_B
        long_sup1=lambd_A
    
    lat_inf.append(lat_inf1)
    lat_sup.append(lat_sup1)
    long_sup.append(long_sup1)
    long_inf.append(long_inf1)    
lat_inf, lat_sup, long_inf, long_sup = np.array(lat_inf),np.array(lat_sup),np.array(long_inf), np.array(long_sup)  

# Opening Database:
DB = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython1\orbite\lunar_crater_database_robbins_2018.csv")

mus = []
N=[]
mus.append(X)
pre=[]
post=[]
pre.append(X)
post.append(X)
#for i in range(len(df)):    
for i in range(15):
    
    ##########################
    # IMPLEMENTAZIONE FILTRO

    if i==0:
        
        (X, P) = kf_predict(X, P, phii, Q) 
        pre.append(X) #Valore del vettore di stato prima della correzione

        # Filtering DATABASE:
        lat_bounds=[lat_inf[0], lat_sup[0]]
        lon_bounds=[long_inf[0],long_sup[0]]
        craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='ROBBINS')

        if craters_cat is None:  
            print ("Non sono stati trovati crateri all'istante ", time[i])
            N.append(0)
        else:
            #Filtro i crateri trovati in base al diametro, vengono scartati tutti quelli con diametro inferiore a 2 metri e 
            #superiore a 25 metri
            indexNames = craters_cat[ (craters_cat['Diam'] <= 2) ].index
            craters_cat.drop(indexNames , inplace=True)
            indexNames = craters_cat[ (craters_cat['Diam'] >= 25) ].index
            craters_cat.drop(indexNames , inplace=True)
            craters_cat = craters_cat.reset_index(drop=True)
            N_crat =len(craters_cat)
            
            if N_crat>5: #se ho più di 5 crateri, ne vengono scelti 5 tra quelli individuati, mentre i restanti vengono scartati
                craters_cat=craters_cat.sample(5)
                craters_cat = craters_cat.reset_index(drop=True)
                N_crat=len(craters_cat)

            else: 
                print(" ") #ho messo un print vuoto perchè continue mi dava problemi  
            
            N.append(N_crat)

            crater_Latitudes, crater_Longitudes = craters_cat['Lat'], craters_cat['Lon']
            print("Sono stati trovati ", N_crat, "crateri all'istante ", time[0])

            x_c, y_c, z_c = [], [], []  #trovo le posizioni dei crateri
            for i_c in range(N_crat):
                altitude = 0
                latitude = crater_Latitudes[i_c]
                longitude = crater_Longitudes[i_c]
                x, y, z = spherical2cartesian(altitude, latitude, longitude)
                x_c.append(x)
                y_c.append(y)
                z_c.append(z)
            x_c, y_c, z_c = np.array(x_c),np.array(y_c),np.array(z_c)
            
            dim_z = 2*N_crat
            
            #Diversi valori per la diagonale della matrice R
            #R=np.eye(dim_z)*156018.59187512
            #R=np.eye(dim_z)*1560186.35585705
            R=np.eye(dim_z)*0.10745133

            #costruisco la matrice H e le misure
            H_allcraters=np.empty([0,12])
            u_vec_tot=np.empty([0,1])
            for i_crat in range(N_crat): 
                xc_i = x_c[i_crat] 
                yc_i = y_c[i_crat]
                zc_i = z_c[i_crat] 
            
                (H_i,u_vec_i)=H_matrix(init_x,init_y, init_z, xc_i,yc_i,zc_i,init_q1,init_q2,init_q3,init_q4) 

                H_allcraters = np.vstack((H_allcraters, H_i))
                u_vec_tot=np.vstack((u_vec_tot, u_vec_i))
        
            HH = H_allcraters
            mis=u_vec_tot
            HH=HH.astype('float64') #necessario altrimenti alcune componenti della matrice risultavano essere degli oggetti
            mis=mis.astype('float64')
            
            (X, P, K, IM, IS) = kf_update(X, P, mis, HH, R)
            post.append(X) #vettore di stato dopo la correzione
            
    else:     
        ########################################################################
        # PREDICTOR:     
  
        ics,ips,zet = real_X[i],real_Y[i], real_Z[i]
        vix,viy,viz = real_Vxs[i],real_Vys[i], real_Vzs[i]
        qqq1,qqq2,qqq3 = real_q1[i],real_q2[i], real_q3[i]
        ome1,ome2,ome3=real_om1[i],real_om2[i], real_om3[i]
        qq4=real_q4[i]

        F = find_F_matrix(ics,ips,zet,qqq1,qqq2,qqq3,qq4,ome1,ome2,ome3)  
        phii = np.eye(12)+F*dt 

        # Predict:
        (X, P) = kf_predict(X, P, phii, Q)
        pre.append(X)

        # Filtering DATABASE:
        lat_bounds=[lat_inf[i], lat_sup[i]]
        lon_bounds=[long_inf[i],long_sup[i]]
        craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='ROBBINS')

        if craters_cat is None:  
            print ("Non sono stati trovati crateri all'istante ", time[i])
            N.append(0)

        else: #se vengono visti crateri, le misure sono le linee di vista di tali crateri 

            indexNames = craters_cat[ (craters_cat['Diam'] <= 2) ].index
            craters_cat.drop(indexNames , inplace=True)
            indexNames = craters_cat[ (craters_cat['Diam'] >= 25) ].index
            craters_cat.drop(indexNames , inplace=True)
            craters_cat = craters_cat.reset_index(drop=True)
          
            N_crat =len(craters_cat)

            if N_crat>5:
                craters_cat=craters_cat.sample(5)
                craters_cat = craters_cat.reset_index(drop=True)
                N_crat=len(craters_cat)

            else: 
                print(" ")

            print(" ") #ho separato i due cicli if/else con un print perchè elif mi dava problemi

            if N_crat==0:
                print("Non sono stati trovati crateri all'istante ", time[i])
                N.append(0)
            else:        
                N.append(N_crat)
                crater_Latitudes, crater_Longitudes = craters_cat['Lat'], craters_cat['Lon']
                print("Sono stati trovati ", N_crat, "crateri all'istante ", time[i])

                x_c, y_c, z_c = [], [], []
                for i_c in range(N_crat):
                    altitude = 0
                    latitude = crater_Latitudes[i_c]
                    longitude = crater_Longitudes[i_c]
                    x, y, z = spherical2cartesian(altitude, latitude, longitude)
                    x_c.append(x)
                    y_c.append(y)
                    z_c.append(z)
                x_c, y_c, z_c = np.array(x_c),np.array(y_c),np.array(z_c)
           
                #Matrice di misura
                dim_z = 2*N_crat #z ha dimensione 2*N_crat identificati  

                #Measurement noise matrix
                #R=np.eye(dim_z)*156018.59187512 
                #R=np.eye(dim_z)*1560186.35585705
                R=np.eye(dim_z)*0.10745133

                H_allcraters=np.empty([0,12])
                u_vec_tot=np.empty([0,1])
                for i_crat in range(N_crat): 
                    xc_i = x_c[i_crat] 
                    yc_i = y_c[i_crat]
                    zc_i = z_c[i_crat] 
                    
                    (H_i,u_vec_i)=H_matrix(ics,ips, zet, xc_i,yc_i,zc_i,qqq1,qqq2,qqq3,qq4) 

                    H_allcraters = np.vstack((H_allcraters, H_i))
                    u_vec_tot=np.vstack((u_vec_tot, u_vec_i))    
        
                HH = H_allcraters
                mis=u_vec_tot
                HH=HH.astype('float64')
                mis=mis.astype('float64')
                (X, P, K, IM, IS) = kf_update(X, P, mis, HH, R)
                post.append(X)
               
    mus.append(X)

mus=np.array(mus)
pre=np.array(pre)
post=np.array(post)
N=np.array(N)

x_pred = []
y_pred = []
z_pred = []
vx_pred = []
vy_pred = []
vz_pred = []
q1_pred = []
q2_pred = []
q3_pred = []
om1_pred = []
om2_pred = []
om3_pred = []

for mu in mus:
    x = mu[0]
    x_pred.append(x)

    y = mu[1]
    y_pred.append(y)
       
    z = mu[2]
    z_pred.append(z)
    
    vx = mu[3]
    vx_pred.append(vx)
    
    vy = mu[4]
    vy_pred.append(vy)
    
    vz = mu[5]
    vz_pred.append(vz)
    
    q1 = mu[6]
    q1_pred.append(q1)
    
    q2 = mu[7]
    q2_pred.append(q2)
    
    q3 = mu[8]
    q3_pred.append(q3)
    
    om1 = mu[9]
    om1_pred.append(om1)
    
    om2 = mu[10]
    om2_pred.append(om2)
    
    om3 = mu[11]
    om3_pred.append(om3)

x_true = real_X[:len(x_pred)]
y_true = real_Y[:len(y_pred)] 
z_true = real_Z[:len(z_pred)]
vx_true = real_Vxs[:len(vx_pred)]
vy_true = real_Vys[:len(vy_pred)]
vz_true = real_Vzs[:len(vz_pred)]
q1_true = real_q1[:len(q1_pred)]
q2_true = real_q2[:len(q2_pred)]
q3_true = real_q3[:len(q3_pred)]
om1_true = real_om1[:len(om1_pred)]
om2_true = real_om2[:len(om2_pred)]
om3_true = real_om3[:len(om3_pred)]

#grafici
lw=1

plt.figure(dpi=150, tight_layout=True)
#plt.figure()
plt.subplot(3,1,1) 
plt.plot(x_pred, '-k', linewidth=lw)
plt.plot(x_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('X [Km]')

plt.subplot(3,1,2) 
plt.plot(y_pred, '-k', linewidth=lw)
plt.plot(y_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Y [Km]')

plt.subplot(3,1,3) 
plt.plot(z_pred, '-k', linewidth=lw)
plt.plot(z_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Z [Km]')
plt.show(block=False)

plt.figure(dpi=150, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(vx_pred, '-k', linewidth=lw)
plt.plot(vx_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vx [km/sec]')

plt.subplot(3,1,2) 
plt.plot(vy_pred, '-k', linewidth=lw)
plt.plot(vy_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vy [km/sec]')

plt.subplot(3,1,3) 
plt.plot(vz_pred, '-k', linewidth=lw)
plt.plot(vz_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vz [km/sec]')
plt.show(block=False)

plt.figure(dpi=150, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(q1_pred, '-k', linewidth=lw)
plt.plot(q1_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Q1')

plt.subplot(3,1,2) 
plt.plot(q2_pred, '-k', linewidth=lw)
plt.plot(q2_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Q2')

plt.subplot(3,1,3) 
plt.plot(q3_pred, '-k', linewidth=lw)
plt.plot(q3_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Q3')
plt.show(block=False)

plt.figure(dpi=150, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(om1_pred, '-k', linewidth=lw)
plt.plot(om1_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 1 [°/sec]')

plt.subplot(3,1,2) 
plt.plot(om2_pred, '-k', linewidth=lw)
plt.plot(om2_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 2 [°/sec]')

plt.subplot(3,1,3) 
plt.plot(om3_pred, '-k', linewidth=lw)
plt.plot(om3_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Omega 3 [°/sec]')
plt.show(block=False)

plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
x_pred = np.array(x_pred)
x_true = np.array(x_true)
diff_x = []
for x,y in zip(x_pred,x_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along X ')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(312)
y_pred = np.array(y_pred)
y_true = np.array(y_true)
diff_y = []
for x,y in zip(y_pred,y_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along Y ')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(313)
z_pred = np.array(z_pred)
z_true = np.array(z_true)
diff_z = []
for x,y in zip(z_pred,z_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along Z ')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km')
plt.show(block=False)


plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
vx_pred = np.array(vx_pred)
vx_true = np.array(vx_true)
diff_x = []
for x,y in zip(vx_pred,vx_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along Vx')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km/sec')

plt.subplot(312)
vy_pred = np.array(vy_pred)
vy_true = np.array(vy_true)
diff_y = []
for x,y in zip(vy_pred,vy_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along Vy')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km/sec')

plt.subplot(313)
vz_pred = np.array(vz_pred)
vz_true = np.array(vz_true)
diff_z = []
for x,y in zip(vz_pred,vz_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along Vz')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km/sec')
plt.show(block=False)


plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
q1_pred = np.array(q1_pred)
q1_true = np.array(q1_true)
diff_x = []
for x,y in zip(q1_pred,q1_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error in Q1')
plt.plot(diff_x, '-k', linewidth=lw)

plt.subplot(312)
q2_pred = np.array(q2_pred)
q2_true = np.array(q2_true)
diff_y = []
for x,y in zip(q2_pred,q2_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error in Q2')
plt.plot(diff_y, '-k', linewidth=lw)

plt.subplot(313)
q3_pred = np.array(q3_pred)
q3_true = np.array(q3_true)
diff_z = []
for x,y in zip(q3_pred,q3_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error in Q3')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.show(block=False)

plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
om1_pred = np.array(om1_pred)
om1_true = np.array(om1_true)
diff_x = []
for x,y in zip(om1_pred,om1_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along omega1')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('deg/sec')

plt.subplot(312)
om2_pred = np.array(om2_pred)
om2_true = np.array(om2_true)
diff_y = []
for x,y in zip(om2_pred,om2_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along omega2')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('deg/sec')

plt.subplot(313)
om3_pred = np.array(om3_pred)
om3_true = np.array(om3_true)
diff_z = []
for x,y in zip(om3_pred,om3_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along omega3')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('deg/sec')
plt.show(block=False)

x_pre = []
y_pre = []
z_pre = []
vx_pre = []
vy_pre = []
vz_pre = []

x_post = []
y_post = []
z_post = []
vx_post = []
vy_post = []
vz_post = []

for pr in pre:
    x = pr[0]
    x_pre.append(x)

    y = pr[1]
    y_pre.append(y)
       
    z = pr[2]
    z_pre.append(z)
    
    vx = pr[3]
    vx_pre.append(vx)
    
    vy = pr[4]
    vy_pre.append(vy)
    
    vz = pr[5]
    vz_pre.append(vz)

for po in post:
    x = po[0]
    x_post.append(x)

    y = po[1]
    y_post.append(y)
       
    z = po[2]
    z_post.append(z)
    
    vx = po[3]
    vx_post.append(vx)
    
    vy = po[4]
    vy_post.append(vy)
    
    vz = po[5]
    vz_post.append(vz)

plt.figure(dpi=150, tight_layout=True)
#plt.figure()
plt.subplot(3,1,1) 
plt.plot(x_pre, '-k', linewidth=lw)
plt.plot(x_post,'g',linewidth=lw)
plt.plot(x_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('X [Km]')
plt.legend(("Predizione","Correzione","Reale"))

plt.subplot(3,1,2) 
plt.plot(y_pre, '-k', linewidth=lw)
plt.plot(y_post,'g',linewidth=lw)
plt.plot(y_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Y [Km]')

plt.subplot(3,1,3) 
plt.plot(z_pre, '-k', linewidth=lw)
plt.plot(z_post,'g',linewidth=lw)
plt.plot(z_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Z [Km]')
plt.show(block=False)

plt.figure(dpi=150, tight_layout=True)
plt.subplot(3,1,1) 
plt.plot(vx_pre, '-k', linewidth=lw)
plt.plot(vx_post,'g',linewidth=lw)
plt.plot(vx_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vx [km/sec]')
plt.legend(("Predizione","Correzione","Reale"))

plt.subplot(3,1,2) 
plt.plot(vy_pre, '-k', linewidth=lw)
plt.plot(vy_post,'g',linewidth=lw)
plt.plot(vy_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vy [km/sec]')

plt.subplot(3,1,3) 
plt.plot(vz_pre, '-k', linewidth=lw)
plt.plot(vz_post,'g',linewidth=lw)
plt.plot(vz_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Vz [km/sec]')
plt.show(block=False)

plt.figure()
plt.title('Numero di crateri individuati')
plt.plot(N, marker="o", linestyle="")
plt.show()
