import numpy as np 
from numpy.linalg import inv, pinv
import math
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.twobody.propagation import cowell as cowell
from poliastro.core.perturbations import J3_perturbation, J2_perturbation
from poliastro.core.propagation import func_twobody
import sympy as sym

#FOV=2.85 #° NARROW ANGLE CAMERA
FOV=61.4 #° WAC

# Funzione per passare da LCLF a ENU
def LCLF2ENU(x, y, z, lat, long):
    phi=math.radians(lat)
    lam=math.radians(long)
    rot_Matrix=np.array([[-np.sin(lam), np.cos(lam), 0], [-np.cos(lam)*np.sin(phi), -np.sin(lam)*np.sin(phi), np.cos(phi)],[np.cos(lam)*np.cos(phi), np.sin(lam)*np.cos(phi), np.sin(phi)]])
    LCLF=np.array([x,y,z])
    E, N, U = np.dot(rot_Matrix,LCLF)
    return np.array(E), np.array(N), np.array(U)

# Funzione per passare da ENU a LCLF
def ENU2LCLF(e, n, u, lat, long):
    phi=math.radians(lat)
    lam=math.radians(long)
    rot_Matrix=np.array([[-np.sin(lam), -np.cos(lam)*np.sin(phi), np.cos(lam)*np.cos(phi)], [np.cos(lam), -np.sin(lam)*np.sin(phi), np.sin(lam)*np.cos(phi)],[0, np.cos(phi), np.sin(phi)]])
    ENU=np.array([e,n,u])
    x, y, z = np.dot(rot_Matrix,ENU)
    return np.array(x), np.array(y), np.array(z)

def SW_nadir(H):
    SW1=2*H*np.tan(math.radians(0.5*FOV))
    SW=SW1*1000 #in metri
    return SW

#########################
#MATRICE PHI
def find_F_matrix(x:float,y:float,z:float,q1:float,q2:float,q3:float,q4:float,omega1:float,omega2:float,omega3:float)-> np.array:
    #Inerzia del satellite
    II1=3000
    II2=2000   #kg*m^2
    II3=1700
    Ip1=(II2-II3)/II1
    Ip2=(II1-II3)/II2
    Ip3=(II1-II2)/II3
    mi = 4.9048695e3 # km^3/s^2 
    I_3x3=np.eye(3)
    Zero_3x3=np.zeros((3,3))
    	
    r = np.sqrt( x**2 + y**2 + z**2)

    J = np.zeros((3,3))
    # First Row
    J[0,0] = ((3*mi*(x**2))/(r**5)) - mi/(r**3)
    J[0,1] = (3*mi*x*y)/(r**5)
    J[0,2] = -(3*mi*x*z)/(r**5) 
    # Second Row
    J[1,0] = (3*mi*y*x)/(r**5)
    J[1,1] = -((3*mi*(y**2))/(r**5)) - mi/(r**3) 
    J[1,2] = -(3*mi*y*z)/(r**5) 
    # Third Row
    J[2,0] = -(3*mi*z*x)/(r**5)
    J[2,1] = -(3*mi*z*y)/(r**5) 
    J[2,2] = -((3*mi*(z**2))/(r**5)) - mi/(r**3) 
    # End

    dfqdq = np.zeros((3,3))
    
    dfqdq[0,0]= -(q1/q4)*(omega1/2)
    dfqdq[0,1]= omega3/2-(q2/q4)*omega1/2
    dfqdq[0,2]= -omega2/2-(q3/q4)*omega1/2

    dfqdq[1,0]= -omega3/2-(q1/q4)*omega2/2
    dfqdq[1,1]= -(q2/q4)*omega2/2
    dfqdq[1,2]= omega1/2-(q3/q4)*omega2/2

    dfqdq[2,0]= omega2/2-(q1/q4)*omega3/2
    dfqdq[2,1]= -omega1/2-(q2/q4)*omega3/2
    dfqdq[2,2]= -(q3/q4)*omega3/2

    dfqdomega = np.zeros((3,3))
    
    dfqdomega[0,0]= q4/2
    dfqdomega[0,1]= -q3/2
    dfqdomega[0,2]= q2/2

    dfqdomega[1,0]= q3/2
    dfqdomega[1,1]= q4/2
    dfqdomega[1,2]= -q1/2

    dfqdomega[2,0]= -q2/2
    dfqdomega[2,1]= q1/2
    dfqdomega[2,2]= q4/2
   
    dfomegadomega = np.zeros((3,3))

    dfomegadomega[0,0]= 0
    dfomegadomega[0,1]= omega3*Ip1
    dfomegadomega[0,2]= omega2*Ip1 

    dfomegadomega[1,0]= -omega3*Ip2
    dfomegadomega[1,1]= 0
    dfomegadomega[1,2]= -omega1*Ip2

    dfomegadomega[2,0]= omega2*Ip3
    dfomegadomega[2,1]= omega1*Ip3
    dfomegadomega[2,2]= 0
 
    # Bulk
    I_3x3_n = np.eye(3)
    I_3x3_n[1,1]=-1
    tmp1 = np.hstack((Zero_3x3, I_3x3_n, Zero_3x3, Zero_3x3)) 
    tmp2 = np.hstack((J,Zero_3x3,Zero_3x3,Zero_3x3))
    tmp3 = np.hstack((Zero_3x3,Zero_3x3,dfqdq,dfqdomega))
    tmp4 = np.hstack((Zero_3x3,Zero_3x3,Zero_3x3,dfomegadomega))

    phi = np.vstack((-tmp1,tmp2,tmp3,tmp4))

    return phi
###################
#FILTRO DI KALMAN

#X: The mean state estimate of the previous step (k-1). 
#P: The state covariance of previous step (k−1). 
#A: The transition nxn matrix. 
#Q: The process noise covariance matrix. 

def kf_predict(X, P, A, Q):     
    X = np.dot(A, X)      
    P = np.dot(A, np.dot(P, A.T)) + Q     
    return(X,P)

#At the time step k, the update step computes the posterior mean X and covariance P of the system state given a new measurement Y. 
# The Python function kf_update performs  the update  of X  and P  giving  the predicted X  and P  matrices, the measurement vector Y, 
# the measurement matrix H and the measurement covariance matrix R. 

def kf_update(X, P, Y, H, R):     
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

#PERTURBATIONS FOR COWELL PROPAGATION
def f(t0, u_, k):
    du_kep = func_twobody(t0, u_, k)
    ax, ay, az = J2_perturbation(
    t0, u_, k, J2=Moon.J2.value, R=Moon.R.to(u.km).value)
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad

##################
#MATRICE DI MISURA H
def H_matrix(ix,iy,iz,xcr,ycr,zcr,iq1,iq2,iq3,iq4):
    S = 0.006 # lunghezza focale WAC
    #VETTORE u
    I_3x3=np.eye(3)
    xx, yy, zz = sym.symbols('xx,yy,zz')              #componenti posizione (vettore r)
    vx, vy, vz = sym.symbols('vx,vy,vz')        #componenti velocita' (vettore V)
    q1, q2, q3, q4 = sym.symbols('q1,q2,q3,q4') #componenti quaternione (vettore q)
    omega1, omega2, omega3= sym.symbols('omega1,omega2,omega3')      #componenti omega
    xc, yc, zc = sym.symbols('xc,yc,zc')        #componenti posizione del cratere (vettore rho)

    X_d=sym.Array([xx,yy,zz,vx,vy,vz,q1,q2,q3,omega1,omega2,omega3]) #vettore di stato

    r = sym.sqrt( xx**2 + yy**2 + zz**2) #|r|
    norm = sym.sqrt( (xx-xc)**2 + (yy-yc)**2 + (zz-zc)**2)   #|r-rho|

    px11 = 0
    px12 = -q3
    px13 = q2 
    px21 = q3
    px22 = 0 
    px23 = -q1
    px31 = -q2
    px32 = q1
    px33 = 0
    px = sym.Matrix(([px11, px12, px13], [px21, px22, px23], [px31, px32, px33]))
    p=sym.Matrix(([q1], [q2], [q3])) #vettore colonna prime tre componenti quaternione
                                                           
    zita1=q4*I_3x3+px
    zita=sym.Matrix(([zita1], [-p.T]))

    psi1=q4*I_3x3-px
    psi=sym.Matrix(([psi1], [-p.T]))

    A=zita.T*psi

    r_vec=sym.Matrix(([xx], [yy], [zz]))   # vettore r  
    rho=sym.Matrix(([xc], [yc], [zc]))     # vettore rho                                                                         

    l=A*((r_vec-rho)/norm)  #los del cratere

    lxi=l[0]
    lyi=l[1]
    lzi=l[2]
    l12=sym.Matrix(([lxi],[lyi])) #prime due componenti del los

    u_vec=(S/lzi)*l12 #measurement function
    Hm=u_vec.jacobian(X_d)
    aH=Hm.subs(xc,xcr).subs(yc,ycr).subs(zc,zcr).subs(xx,ix).subs(yy,iy).subs(zz,iz).subs(q1,iq1).subs(q2,iq2).subs(q3,iq3).subs(q4,iq4)
    u_vet=u_vec.subs(xc,xcr).subs(yc,ycr).subs(zc,zcr).subs(xx,ix).subs(yy,iy).subs(zz,iz).subs(q1,iq1).subs(q2,iq2).subs(q3,iq3).subs(q4,iq4)
    return aH,u_vet