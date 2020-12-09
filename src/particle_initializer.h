#pragma once

#include <cmath>

#include "particle_simulation.h"

//Includes functions that create particle structs for the initilization of
//the ParticleSimulation class

template<typename T>
struct InPutParticle {
  Particle<T> particle;
  T m;
};

//These three functions apply rotation matrices to the 3D input vector
template<typename T>
void RotateArroundX_Achis(T* vector, T angle){
  T tmp[3] = {0,0,0};
  tmp[1] = vector[1]*cos(angle) - vector[2]*sin(angle);
  tmp[2] = vector[1]*sin(angle) + vector[2]*cos(angle);
  vector[1] = tmp[1];
  vector[2] = tmp[2];
}

template<typename T>
void RotateArroundY_Achis(T* vector, T angle){
  T tmp[3] = {0,0,0};
  tmp[0] = vector[0]*cos(angle) + vector[2]*sin(angle);
  tmp[2] = (-1)*vector[0]*sin(angle) + vector[2]*cos(angle) ;
  vector[0] = tmp[0];
  vector[2] = tmp[2];
}

template<typename T>
void RotateArroundZ_Achis(T* vector, T angle){
  T tmp[3] = {0,0,0};
  tmp[0] = vector[0]*cos(angle) - vector[1]*sin(angle);
  tmp[1] = vector[0]*sin(angle) + vector[1]*cos(angle);
  vector[0] = tmp[0];
  vector[1] = tmp[1];
}

//default mass = earthmass
template<typename T>
InPutParticle<T> BodyAtCenter(T mass=5.974e+24){
  InPutParticle<T> my_particle;
  for(int i=0; i<3; i++){
    my_particle.particle.r[i] = 0;
    my_particle.particle.v[i] = 0;
  }
  my_particle.m = mass;
  return my_particle;
}

//Assumes central-body_mass >> moon mass for calculation of velocity.
//Default parameters are for THE moon in its real orbit, in the XY-plane with
//phase = 0.
template<typename T>
InPutParticle<T> CircularOrbitMoon(T orbital_phase_z = 0.0,
                                T orbital_plane_angle_polar_y = 0.0,
                                T orbital_plane_angle_azimutal_x = 0.0,
                                bool clockwise = true,
                                T central_body_mass = 5.974e+24,
                                T moon_mass = 7.342e+22,
                                T orbital_radius = 3.844e+8){
  T G = 6.67430e-11;  // gravitational constant
  T x[3];
  T v[3];
  T velocity = sqrt((G*central_body_mass)/orbital_radius);

  //For the sake of simplicity the moon position and velocity is first set to
  //one in the XY plane with phase = 0 and then subsequently rotated to fulfill
  //the specified parameters for the orbital phase and the orbital plane angles.
  x[0] = orbital_radius;
  x[1] = 0;
  x[2] = 0;

  v[0] = 0;
  if(clockwise){
    v[1] = velocity;
  } else {
    v[1] = -velocity;
  }
  v[2] = 0;

  //Adjusting moon parameter to specified orrbital_plane_angle_azimutal.
  if (orbital_plane_angle_azimutal_x != 0.0){
    RotateArroundX_Achis(x, orbital_plane_angle_azimutal_x);
    RotateArroundX_Achis(v, orbital_plane_angle_azimutal_x);
  }

  //Adjusting moon parameter to specified orrbital_plane_angle_polar.
  if (orbital_plane_angle_polar_y != 0.0){
    RotateArroundY_Achis(x, orbital_plane_angle_polar_y);
    RotateArroundY_Achis(v, orbital_plane_angle_polar_y);
  }

  //Adjusting moon parameter to specified orbital_phase_z_rot_z.
  if (orbital_phase_z != 0.0){
    RotateArroundZ_Achis(x, orbital_phase_z);
    RotateArroundZ_Achis(v, orbital_phase_z);
  }

  InPutParticle<T> my_particle;
  for(int i=0; i<3; i++){
    my_particle.particle.r[i] = x[i];
    my_particle.particle.v[i] = v[i];
  }
  my_particle.m = moon_mass;
  return my_particle;
}

//Like CircularOrbitMoon() but adds the possibility to add a deviation of the
//position and velocity from an in-plane-circular orbit.
//A typical input for the x_deviation vector would be a rondom vector that is
//e.g normaly distributed + different orbital phases --> creates ring of moons.
//Deviations of the velocity create elliptical orbits (for y_deviation[1])
//otherwise rotate the orbital plane.
template<typename T>
InPutParticle<T> NonCircularOrbitMoon(T* x_deviation,
                                   T* v_deviation,
                                   T orbital_phase_z = 0.0,
                                   T orbital_plane_angle_polar_y = 0.0,
                                   T orbital_plane_angle_azimutal_x = 0.0,
                                   bool clockwise = true,
                                   T central_body_mass = 5.974e+24,
                                   T moon_mass = 7.342e+22,
                                   T orbital_radius = 3.844e+8){
  T G = 6.67430e-11;  // gravitational constant
  T x[3];
  T v[3];
  T velocity = sqrt((G*central_body_mass)/orbital_radius);

  //For the sake of simplicity the moon position and velocity is first set to
  //one in the XY plane with phase = 0 and then subsequently rotated to fulfill
  //the specified parameters for the orbital phase and the orbital plane angles.
  x[0] = orbital_radius + x_deviation[0];
  x[1] = 0 + x_deviation[1];
  x[2] = 0 + x_deviation[2];

  v[0] = 0 + v_deviation[0];
  if(clockwise){
    v[1] = velocity + v_deviation[1];
  } else {
    v[1] = -velocity + v_deviation[1];
  }
  v[2] = 0 + v_deviation[2];

  //Adjusting moon parameter to specified orrbital_plane_angle_azimutal.
  if (orbital_plane_angle_azimutal_x != 0.0){
    RotateArroundX_Achis(x, orbital_plane_angle_azimutal_x);
    RotateArroundX_Achis(v, orbital_plane_angle_azimutal_x);
  }

  //Adjusting moon parameter to specified orrbital_plane_angle_polar.
  if (orbital_plane_angle_polar_y != 0.0){
    RotateArroundY_Achis(x, orbital_plane_angle_polar_y);
    RotateArroundY_Achis(v, orbital_plane_angle_polar_y);
  }

  //Adjusting moon parameter to specified orbital_phase_z_rot_z.
  if (orbital_phase_z != 0.0){
    RotateArroundZ_Achis(x, orbital_phase_z);
    RotateArroundZ_Achis(v, orbital_phase_z);
  }

  InPutParticle<T> my_particle;
  for(int i=0; i<3; i++){
    my_particle.particle.r[i] = x[i];
    my_particle.particle.v[i] = v[i];
  }
  my_particle.m = moon_mass;
  return my_particle;
}
