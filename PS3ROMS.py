import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


### FUNCTIONS ###

def rHat(vec):
    """This function takes in the radius in vector form   arrow(r) = (x, y, z)
    and returns the directional vector   hat(r) = arrow(r)/|arrow(r)|.
    The purpose of this is to turn any scalar in the r-direction into its x,y,z components."""
    if np.sum(vec)==0:
        return 0.0
    else:
        return vec / np.linalg.norm(vec)


def accG(r1, m_list, r_list):
    """Calculates the acceleration due to the Sun and all other planets (except the current object itself)."""
    accel = -G*m_list/(r1-r_list)**2
    return np.sum(accel, where=(accel!=-np.inf))


def check_for_crash(current_num, positions, m_list, r_list):
    """Checks if two planets come between a certain radius. If they do, the smaller planet is moved
    far away so that it does not interfere with the simulation and its mass is added to the larger planet."""
    where_too_close = (0.000477894503*1.5 > np.abs(r_list[current_num]-r_list)) & (m_list!=m_list[current_num])
    mass_place = np.where(where_too_close==True)[0]
    
    if mass_place:
        mass_difference = np.sum(m_list[where_too_close]) - m_list[current_num]
        
        if mass_difference<0.0:
            positions[mass_place,:] = np.array([9999,9999,9999])
            m_list[mass_place] += m_list[current_num]
            
        elif mass_difference>0.0:
            positions[current_num,:] = np.array([9999,9999,9999])
            m_list[current_num]  += m_list[mass_place]
            
    return positions, m_list


### User Inputs ###

is_cromer = True        # for Euler-Cromer method, False for Euler
is_solar_system = True  # if not, then use random planets
num_planets = 0         # extra planets
total_time = 500        # yr
timestep = 0.01         # yr
jup2r = 2000.0           # AU


### Global Constants ###

G = 39.4                      # AU^3/MSUN/yr^2
MSUN = 1.0                    # MSUN
MJUP = 1.898e27 / 1.989e30    # MSUN
MEARTH = 5.972e24 / 1.989e30  # MSUN


### PS3-ROMS ###

if is_solar_system==True:
    num_planets = 7
    num_obj = num_planets + 3
else:
    num_obj = num_planets + 3

    
# Create vectors for position, velocity, acceleration
pos = np.zeros(shape=(num_obj, 3))  # 3 columns: x,   y,   z
vel = np.zeros(shape=(num_obj, 3))  # 3 columns: v_x, v_y, v_z
acc = np.zeros(shape=(num_obj, 3))  # 3 columns: a_x, a_y, a_z
mass = np.zeros(shape=(num_obj))
names = ['sun', 'jupnew', 'jupiter']

# Define positions/masses of the Sun & two Jupiter-like planets
pos[0:3,0] = np.array([0, jup2r, 5.2])
mass[0:3] = np.array([MSUN, MJUP+0.00001*MJUP, MJUP])
pos[1:3,2] = np.tan(np.radians(np.array([1.3,1.3]))) * pos[1:3,0]

if is_solar_system==True:
    pos[3:,0] = np.array([0.4, 0.7, 1.0, 1.5, 9.6, 19.2, 30.0])
    mass[3:] = np.array([0.0553, 0.815, 1.0, 0.1074, 95.2, 14.54, 17.15]) * MEARTH
    names += ['mercury', 'venus', 'earth', 'mars', 'saturn', 'uranus', 'neptune']
    pos[3:,2] = np.tan(np.radians(np.array([7., 3.4, 0.0, 1.84, 3.4, 0.77, 1.77]))) * pos[3:,0]
else:
    # Start systems at random positions, at Keplarian velocity in the y direction
    pos[3:,0] = np.random.default_rng().uniform(low=0.5, high=30.0, size=num_planets)
    mass[3:] = np.random.default_rng().uniform(low=1.0, high=5.0, size=num_planets) * MEARTH
    pos[3:,2] = np.tan(np.radians(np.random.default_rng().uniform(low=-3., high=3., size=num_planets))) * pos[3:,0]

vel[1:,1] = np.sqrt(G * MSUN / pos[1:,0])

# Set up arrays to save the data to
num_timesteps = int(total_time/timestep)
pos_save = np.zeros(shape=(num_obj, 3, num_timesteps+1))
vel_save = np.zeros(shape=(num_obj, 3, num_timesteps+1))
pos_save[:,:,0] = np.copy(pos)
vel_save[:,:,0] = np.copy(vel)

for i in range(num_timesteps):
    radii = np.sqrt(np.sum(pos**2, axis=1))
    
    for j in range(len(radii)):
        pos, mass = check_for_crash(j, pos, mass, radii)
        acc[j,:] = rHat(pos[j,:]) * accG(radii[j], mass, radii)

    if is_cromer==True:
        vel += acc*timestep
        pos += vel*timestep
    else:
        pos += vel*timestep
        vel += acc*timestep

    vel_save[:,:,i+1] = np.copy(vel)
    pos_save[:,:,i+1] = np.copy(pos)



### License ###

"""Copyright 2020 Isaiah Tristan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""