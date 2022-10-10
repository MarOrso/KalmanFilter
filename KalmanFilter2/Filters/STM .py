import numpy as np  
from poliastro.bodies import Moon 
from poliastro.twobody.orbit import Orbit
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.core.angles import nu_to_E, E_to_M
import sympy as sym

global MU
global Ip1, Ip2, Ip3

from .CONFIG import *

def Keplerian(pos, vel, att, rot, dt):

     def phi_motion(xx,yy,zz, as_numpy=True):
          x, y, z = sym.symbols('x,y,z')              # componenti posizione (vettore r)
          mi = sym.symbols(r'\mu')             
          r = sym.sqrt(x**2 + y**2 + z**2)

          f00 = ((3*mi*(x**2))/(r**5)) - mi/(r**3)
          f01 = (3*mi*x*y)/(r**5)
          f02 = (3*mi*x*z)/(r**5) 

          f10 = (3*mi*y*x)/(r**5)
          f11 = ((3*mi*(y**2))/(r**5)) - mi/(r**3) 
          f12 = (3*mi*y*z)/(r**5) 

          f20 = (3*mi*x*z)/(r**5)
          f21 = (3*mi*z*y)/(r**5) 
          f22 = ((3*mi*(z**2))/(r**5)) - mi/(r**3) 
          motion = sym.Matrix(([f00, f01, f02], [f10, f11, f12], [f20, f21, f22]))

          phi = sym.Matrix([[sym.zeros(3,3),sym.eye(3,3)], [motion, sym.zeros(3,3)]])
          phi = phi.subs(x,xx).subs(y,yy).subs(z,zz).subs(mi,MU)
          phi = phi.evalf(maxn=300)
          if as_numpy:
               phi = np.array(phi).astype(np.float64)
          return phi

     def phi_attitude(oomega1,oomega2,oomega3, as_numpy=True):

          
          omega1, omega2, omega3 = sym.symbols('omega_1,omega_2,omega_3')              # componenti posizione (vettore r)

          f93 = 0
          f94 = omega3*Ip1
          f95 = omega2*Ip1 

          f103 = -omega3*Ip2
          f104 = 0
          f105 = -omega1*Ip2

          f113 = omega2*Ip3
          f114 = omega1*Ip3
          f115 = 0
          att = sym.Matrix(([f93, f94, f95], [f103, f104, f105], [f113, f114, f115]))
          phi = sym.Matrix(([sym.zeros(3,3), sym.eye(3,3)], [sym.zeros(3,3), att]))
          phi = phi.subs(omega1,oomega1).subs(omega2,oomega2).subs(omega3,oomega3)
          phi = phi.evalf(maxn=300)
          if as_numpy:
               phi = np.array(phi).astype(np.float64)
          return phi

     x,y,z = pos
     vx,vy,vz = vel
     teta1,teta2,teta3 = att #pitch, yaw, roll
     omega1,omega2,omega3 = rot

     motion = phi_motion(x,y,z)
     attitude = phi_attitude(omega1,omega2,omega3)

     # Bulk
     top = np.hstack((motion, np.zeros((6,6)))) 
     bottom = np.hstack((np.zeros((6,6)), attitude))
     phi = np.vstack((top,bottom))

     F = np.eye(12) + phi*dt 
     return F.astype(np.float64)






def Goodyear(r0,v0,dt=10, Verbose=False):

     def M_fg_fdot_gdot(x,y,z,xdot,ydot,zdot, dt=10,Verbose=True):

          def Kepler_NR(M, e, tol=1e-12):
          # Solve Kepler equation with NR method:
               En = M # starting
               while (En - e*np.sin(En) - M > tol):
                    En = En - (En - e*np.sin(En) - M)/(1-e*np.cos(En))
               return En

          def convert_angle_to_0_2pi(a):
          # Auxiliary function:
               while a < 0:
                    a += 2 * np.pi
               while a > 2 * np.pi:
                    a -= 2 * np.pi
               return a


          r0 = np.sqrt(x**2 + y**2 + z**2)
          v0 = np.sqrt(xdot**2 + ydot**2 + zdot**2)
          r = np.hstack([x,y,z]) *u.km
          v = np.hstack([xdot,ydot,zdot])*u.km/u.s

          orbit = Orbit.from_vectors(attractor=Moon, r=r, v=v)
          a = orbit.a.to_value()
          ecc = orbit.ecc.to_value() # eccentricity
          nu0 = orbit.nu.to_value() # true anomaly
          n = orbit.n.to_value() # mean motion
          h0 = orbit.h_mag.to_value()
          if Verbose:
               print('Semi-major axis:',a)
               print('eccentricity:',ecc)
               print('Initial True Anomaly:',nu0)
               print('mean motion n:', n)
               print('Specific angular momentum:',h0)


          E0 = nu_to_E(nu0, ecc)
          if Verbose:
               print('Initial eccentric Anomaly:',E0)
          # mean anomalies for the initial and propagated orbits:
          # M0 = E0 - (h0/np.sqrt(MU*a))
          M0 = E_to_M(E0, ecc)
          if Verbose:
               print('Initial Mean Anomaly:',M0)
          M0 = convert_angle_to_0_2pi(M0)
          if Verbose:
               print('Converted Initial Mean Anomaly:',M0)
          M = n*dt + M0
          if Verbose:
               print('Propagated Mean Anomaly:', M)
          
          M = convert_angle_to_0_2pi(M)
          if Verbose:
               print('Converted Propagated Mean Anomaly:', M)

          # Kepler’s equation is solved from an initial guess based on an approximation series, 
          # then iterated by Newton-Raphson’s method, until convergence to the level of 10−12 is achieved.
          # M = E - e np.sin(E) --> E - e sin(E) - M = 0
          E = Kepler_NR(M,ecc)
          if Verbose:
               print('Eccentric Propagated Mean Anomaly:', M)
          # variation of the eccentric anomaly
          deltaE = E - E0
          if Verbose:
               print('deltaE:', deltaE)
          deltaE = convert_angle_to_0_2pi(deltaE)
          if Verbose:
               print('deltaE Converted:', deltaE)

          # transcendental functions Goodyear:
          s0 = np.cos(deltaE)
          s1 = np.sqrt(a/MU) * np.sin(deltaE)
          s2 = (a/MU)*(1-s0)
          if Verbose:
               print('Transcendental Function:', s0, s1, s2)
          # the function f,g,fdot,gdot:
          r = r0*s0 + h0*s1 + MU*s2

          f = 1 - (MU*s2)/r0
          g = r0 * s1 + h0 * s2
          fdot = -(MU*s1)/(r*r0)
          gdot = 1 - (MU*s2/r)
          if Verbose:
               print('f:',f)
               print('g:',g) 
               print('fdot:',fdot)
               print('gdot:',gdot)

          # propagate state:
          assert np.isfinite(f)
          assert np.isfinite(g)
          assert np.isfinite(fdot)
          assert np.isfinite(gdot)
          r, v = r0*f + v0*g, r0*fdot + v0*gdot

          # secular component:
          deltaE = deltaE + int(dt*n/(2*np.pi))*2*np.pi
          s4p = np.cos(deltaE) - 1 
          s5p = np.sin(deltaE) - deltaE
          U = s2*dt + np.sqrt((a/MU)**5) * (deltaE*s4p -3*s5p)

          M11 = ((s0/(r*r0)) + 1/(r0**2) +1/(r**2))*fdot - MU**2 * (U/(r**3 * r0**3))
          M12 = (fdot*s1/r + (gdot-1)/r**2 )
          M13 = ((gdot-1)*s1)/r - MU*U/r**3 

          M21 = -(fdot*s1/r0 + (f-1)/r0**2)
          M22 = -(fdot*s2)
          M23 = -(gdot-1)*s2

          M31 = (f-1)*s1/r0 - MU*U/r0**3
          M32 = (f-1)*s2 
          M33 = g*s2 - U
          row1 = np.hstack([M11,M12,M13])
          row2 = np.hstack([M21,M22,M23])
          row3 = np.hstack([M31,M32,M33])

          orbit=orbit.propagate(value=10*u.s)
          rr = orbit.r.to_value()
          vv = orbit.v.to_value()

          M = np.vstack((row1,row2,row3))
          return M, [rr,vv], f,g,fdot,gdot

     # Prompt state transition orbital matrix:
     x,y,z = r0
     r0 = np.array([x,y,z])
     xdot,ydot,zdot = v0
     v0 = np.array([xdot,ydot,zdot])
     M, [r,v], f,g,fdot,gdot = M_fg_fdot_gdot(x,y,z,xdot,ydot,zdot, dt)
     if Verbose:
          print('M, and state calculation performed successfully.')

     # State transition matrix calculation:
     r0=np.hstack(r0)
     v0=np.hstack(v0)
     tmp = np.vstack((r0,v0))

     phi11 = f*np.eye(3) +  np.vstack((r,v)).T @ np.array([[M[1,0], M[1,1]],[M[2,0], M[2,1]]]) @ tmp  
     phi12 = g*np.eye(3) +  np.vstack((r,v)).T @ np.array([[M[1,1], M[1,2]],[M[2,1], M[2,2]]]) @ tmp
     phi21 = fdot*np.eye(3)+np.vstack((r,v)).T @ np.array([[M[0,0], M[0,1]],[M[1,0], M[1,1]]]) @ tmp
     phi22 = gdot*np.eye(3)+np.vstack((r,v)).T @ np.array([[M[0,1], M[0,2]],[M[1,1], M[1,2]]]) @ tmp

     # r, v = r0*f + v0*g, r0*fdot + v0*gdot

     phi = np.vstack((np.hstack((phi11,phi12)),np.hstack((phi21,phi22))))
     return phi, [r,v]
