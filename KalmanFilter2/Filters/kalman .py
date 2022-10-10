import numpy as np  
import pandas as pd
from numpy.linalg import inv, pinv
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import sympy as sym
from astropy.coordinates.funcs import spherical_to_cartesian, cartesian_to_spherical
from poliastro.twobody.orbit import Orbit
from poliastro.plotting import OrbitPlotter3D
from poliastro.bodies import Earth, Sun, Moon
from poliastro.constants import J2000
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import pickle 
# More info: https://plotly.com/python/renderers/
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected"

global MU
global Ip1, Ip2, Ip3

from .STM import Keplerian
from .CONFIG import *


class KalmanFilter:
     """
     Class for implementing Kalman Filter in python.
     It uses the standard equations of the Kalman Filter, as developed by Rudolph Kalman.
     """

     def __init__(self, position, velocity, attitude, rotation):
          self.x, self.y, self.z = position
          self.vx, self.vy, self.vz = velocity
          self.teta1, self.teta2, self.teta3 = attitude #pitch, yaw, roll
          self.omega1, self.omega2, self.omega3 = rotation
          self.X = [self.x, self.y, self.z,self.vx, self.vy, self.vz, self.teta1, self.teta2, self.teta3, self.omega1, self.omega2, self.omega3]
          
          # Other params:
          self.__dt = dt # Integration time
          self.dt = self.__dt

          #Inertia of satellite (kg*m^2)
          self.II1 = II1
          self.II2 = II2   
          self.II3 = II3

          self.mi = MU # km^3/s^2 
          self.S = S # m (focal length 700 mm per la wide angle camera)
          self.FOV=FOV #° WIDE ANGLE CAMERA
          self.SPAN = SPAN # Searching area for catalog retrieval TODO: change this with a func
          
          self.init_covariance()
          self.checkpoint = {'state':[]}
          self.save()
          print('Kalman Filter properly initialized.')

     def init_covariance(self):
          self.P=np.diag([3.65996971, 0.29041806, 6, 5.52247218, -18.90088871, 0.23225206, 2.8019472, 2.73023895, 3.0819458, 3.35880549, 1.62408328, 1.5544467]).astype(np.float64)

     def state_transition(self):
          """
          Computes the state transition matrix F.
          """

          p,v = self.X[:3], self.X[3:6]
          att, rot = self.X[6:9], self.X[9:12]  
          self.F = Keplerian(p,v,att,rot, dt=self.__dt)
          return self.F

     def orbitPlotter(self, epoch=None):
          frame = OrbitPlotter3D()
          frame.set_attractor(Moon)

          r = np.hstack([self.X[0],self.X[1],self.X[2]]) * u.km
          v = np.hstack([self.X[3],self.X[4],self.X[5]])*u.km/u.s
          orbit = Orbit.from_vectors(attractor=Moon, r=r, v=v)
          self.orbit = orbit
          if epoch is not None:
               orbit.plot(epoch)
          else:
               orbit.plot(label="FederNetV2")
          plt.show()
          # frame.plot_body_orbit(Earth, J2000)


     def row_H(self, xcr,ycr,zcr):
          """
          Prompts the i-th row of measuremnt matrix.
          """
          ix,iy,iz = self.X[0],self.X[1],self.X[2]
          t1,t2,t3 = self.X[6],self.X[7],self.X[8]
          S = self.S # lunghezza focale WAC
          #VETTORE u
          I_3x3=np.eye(3)
          xx, yy, zz = sym.symbols('xx,yy,zz')              #componenti posizione (vettore r)
          vx, vy, vz = sym.symbols('vx,vy,vz')        #componenti velocita' (vettore V)
          teta1,teta2,teta3 = sym.symbols('teta1,teta2,teta3') #angoli di eulero
          omega1, omega2, omega3= sym.symbols('omega1,omega2,omega3')      #componenti omega
          xc, yc, zc = sym.symbols('xc,yc,zc')        #componenti posizione del cratere (vettore rho)

          X_d=sym.Array([xx,yy,zz,vx,vy,vz,teta1,teta2,teta3,omega1,omega2,omega3]) #vettore di stato

          r = sym.sqrt( xx**2 + yy**2 + zz**2) #|r|
          norm = sym.sqrt( (xx-xc)**2 + (yy-yc)**2 + (zz-zc)**2)   #|r-rho|

          A00 = sym.cos(teta2)*sym.cos(teta3)
          A01 = sym.cos(teta3)*sym.sin(teta1)*sym.sin(teta2)+sym.cos(teta1)*sym.sin(teta3)
          A02 = sym.sin(teta1)*sym.sin(teta3)-sym.cos(teta1)*sym.cos(teta3)*sym.sin(teta2)

          A10 = -sym.cos(teta2)*sym.sin(teta3)
          A11 = sym.cos(teta1)*sym.cos(teta3)-sym.sin(teta1)*sym.sin(teta2)*sym.sin(teta3)
          A12 = sym.cos(teta1)*sym.sin(teta2)*sym.sin(teta3)+sym.cos(teta3)*sym.sin(teta1)

          A20 = sym.sin(teta2)
          A21 = -sym.cos(teta2)*sym.sin(teta1)
          A22 = sym.cos(teta2)*sym.cos(teta3)

          A = sym.Matrix(([A00, A01, A02], [A10, A11, A12], [A20, A21, A22]))

          r_vec=sym.Matrix(([xx], [yy], [zz]))   # vettore r  
          rho=sym.Matrix(([xc], [yc], [zc]))     # vettore rho                                                                         

          l=A*((r_vec-rho)/norm)  #los del cratere

          lxi=l[0]
          lyi=l[1]
          lzi=l[2]
          l12=sym.Matrix(([lxi],[lyi])) #prime due componenti del los

          u_vec=(S/lzi)*l12 #measurement function
          Hm=u_vec.jacobian(X_d)
          aH=Hm.subs(xc,xcr).subs(yc,ycr).subs(zc,zcr).subs(xx,ix).subs(yy,iy).subs(zz,iz).subs(teta1,t1).subs(teta2,t2).subs(teta3,t3)
          u_vet=u_vec.subs(xc,xcr).subs(yc,ycr).subs(zc,zcr).subs(xx,ix).subs(yy,iy).subs(zz,iz).subs(teta1,t1).subs(teta2,t2).subs(teta3,t3)
          return aH,u_vet

     def measurement_matrix(self):
          """
          Prompts the measurement matrix H and the u vector.
          """
          H_rows, u_vec_rows=[],[]
          for idx, row in self.craters:
               altitude, latitude, longitude = 0, row['Lat'],row['Lon']
               xc, yc, zc = spherical2cartesian(altitude, latitude, longitude)
               H_i, u_vec_i=self.row_H(xc,yc,zc) 
               H_rows.append(H_i)
               u_vec_rows.append(u_vec_i)

          H = np.vstack(H_rows)
          u_vec=np.vstack(u_vec_rows)
          
          self.H = H
          self.u_vec = u_vec
          return H, u_vec
          
     def save(self):
          self.checkpoint['state'].append(self.X.copy())
          with open('results.pkl', 'wb') as handle:
               pickle.dump(self.checkpoint['state'], handle, protocol=pickle.HIGHEST_PROTOCOL)

     def cowell_propagation(self, prompt_kep=False):
          r = np.hstack([self.X[0],self.X[1],self.X[2]]) * u.km
          v = np.hstack([self.X[3],self.X[4],self.X[5]])*u.km/u.s
          orbit = Orbit.from_vectors(attractor=Moon, r=r, v=v)
          if prompt_kep:
               self.a = orbit.a.to_value()
               self.ecc = orbit.ecc.to_value() # eccentricity
               self.nu0 = orbit.nu.to_value() # true anomaly
               self.n = orbit.n.to_value() # mean motion
               self.h0 = orbit.h_mag.to_value()
          
          orbit=orbit.propagate(value=self.__dt*u.s)
          self.X[0],self.X[1],self.X[2] = orbit.r.to_value()
          self.X[3],self.X[4],self.X[5] = orbit.v.to_value()

     def predict(self): 
          """
          Performs the prediction step of the Kalman Filter.
          """

          q = Q_discrete_white_noise(dim=3, dt=self.__dt, var=0.001)
          Q = block_diag(q, q, q, q)

          self.cowell_propagation()          
          F = self.state_transition()
          self.P = F @ self.P @ self.F.T + Q
          self.save()     
          return self.X, self.P

     def do_measurement(self, DB: pd.DataFrame):
          """
          Performs the measurement of the craters inside the catalogue.
               Input: 
                    DB: pd.DataFrame containing the craters detected and matched.
          """
          # Filtering DATABASE:
          x,y,z = self.X[0].item(), self.X[1].item(), self.X[2].item()
          h,lat,lon = cartesian2spherical(x, y, z)
          lat_bounds=[lat-self.SPAN, lat+self.SPAN]
          lon_bounds=[lon-self.SPAN,lon+self.SPAN]
          craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='ROBBINS')
          self.craters = craters_cat
          return craters_cat

     def update(self, Y, H, R):   
          """
          Performs the prediction step of the Kalman Filter.
          """
          H = self.measurement_matrix()

          IM = np.dot(H, X)  
          PHT = np.dot(P, H.T)
          S = np.dot(H, PHT) + R
          IS = inv(S)
          # 1) Kalman gain:
          K = np.dot(PHT, IS)     
          # 2) Update estimate with measurement:
          X = X + np.dot(K, (Y - np.dot(H, X)))     
          # 3) Update the estimate uncertainty:
          P = P - np.dot(K, np.dot(IS, K.T))         
          return (X,P,K,IM,IS)



################## UTILITY
def adjust_mat(mat):
     mat = np.where(abs(mat) < 1e-12,0, mat)
     mat = np.clip(mat, -50,50)
     mat = np.where(np.isnan(mat),0,mat)
     return mat


def cartesian2spherical(x, y, z):
     """
     Inputs: 
          x, y, z
     Outputs:
          h: Altitude (km)
          Lat: Latitude (deg)
          Lon: Longitude (deg)
     """
     h, Lat, Lon = cartesian_to_spherical(-x, -y, -z)
     R_moon = 1737.4
     h = h - R_moon
     Lon = np.where(Lon.deg < 180, Lon.deg, Lon.deg - 360)
     Lat = np.where(Lat.deg < 90, Lat.deg, Lat.deg - 360)
     # Checking if is single valued:
     if Lon.size == 1:
          h, Lat, Lon = h.item(), Lat.item(), Lon.item()
     return h, Lat, Lon

def spherical2cartesian(h, Lat, Lon):
     """
     Inputs:
          h: Altitude (km)
          Lat: Latitude (deg)
          Lon: Longitude (deg)
     Outputs: 
          x, y, z
     """
     R_moon = 1737.4
     x, y, z = spherical_to_cartesian(
          h + R_moon, np.deg2rad(Lat), np.deg2rad(Lon))
     return -x, -y, -z

def CatalogSearch(H, lat_bounds: np.array, lon_bounds: np.array, CAT_NAME):
    # -180 to 180 // formulation 1
    #   0  to 360 // formulation 2
    # Example: 190 lon //formulation 2 --> -170 lon // formulation 1
    # -10 lon == 350 lon
    # We want to pass from f1 --> f2
    if CAT_NAME == "LROC":
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diameter (km)"])
        LONs = np.array(H["Long"])

    elif CAT_NAME == "HEAD":
        LONs = np.array(H["Lon"])
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diam_km"])
    elif CAT_NAME == "ROBBINS":
        LONs = np.array(H["LON_CIRC_IMG"])
        LATs = np.array(H["LAT_CIRC_IMG"])
        DIAMs = np.array(H["DIAM_CIRC_IMG"])

    elif CAT_NAME == "COMBINED":
        LONs = np.array(H["lon"])
        LATs = np.array(H["lat"])
        DIAMs = np.array(H["diam"])

    LONs_f1 = np.where(LONs > 180, LONs - 360, LONs)

    cond1 = LONs_f1 < lon_bounds[1]
    cond2 = LONs_f1 > lon_bounds[0]
    cond3 = LATs > lat_bounds[0]
    cond4 = LATs < lat_bounds[1]

    filt = cond1 & cond2 & cond3 & cond4

    LATs = LATs[filt]
    LONs_f1 = LONs_f1[filt]
    DIAMs = DIAMs[filt]
    if LONs_f1 != []:
        craters = np.hstack(
            [np.vstack(LONs_f1), np.vstack(LATs), np.vstack(DIAMs)])
        df = pd.DataFrame(data=craters, columns=["Lon", "Lat", "Diam"])
        return df
    else:
        pass

if __name__ == "__main__":
     # Example of usage:
     position = [0,0,0]
     velocity = [0.1,0,0]
     attitude = [0,0,0,0]
     rotation = [0,0,0]

     K_R = 3000

     KF = KalmanFilter(position, velocity, attitude, rotation) 
     
     KF.kf_predict()
     craters=KF.do_measurement()

     if craters is not None:
          craters=craters[craters['Diam'] >= 2 and craters['Diam'] <=25]
          x_c, y_c, z_c = [], [], []
          for idx, row in craters.iterrows():
               altitude, latitude, longitude= 0, row['Lat'],row['Lon']
               x, y, z = spherical2cartesian(altitude, latitude, longitude)
               x_c.append(x)
               y_c.append(y)
               z_c.append(z)
          
          x_c, y_c, z_c = np.array(x_c),np.array(y_c),np.array(z_c)
          dim_z = 2*len(craters)

          # Diversi valori per la diagonale della matrice R
          R=np.eye(dim_z)*K_R



