from poliastro.bodies import Moon 
from astropy import units as u

MU = Moon.k.to(u.km**3/u.s**2)
MU = MU.to_value()

dt = 10
# mi = 4.9048695e3 # km^3/s^2 
S = 0.006 # m (focal length 700 mm per la wide angle camera)
FOV = 61.4 #° WIDE ANGLE CAMERA
SPAN = 3.29 / 2 # Searching area for catalog retrieval

# Satellite:
II1 = 3000
II2 = 2000   
II3 = 1700
# prompting:
Ip1=(II2-II3)/II1
Ip2=(II1-II3)/II2
Ip3=(II1-II2)/II3