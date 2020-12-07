#include <iostream>
#include <thread>
#include <random>

#include "particle_simulation.h"
#include "particle_initializer.h"

//example file intended to show how the simulation is set up and performed
//Example is the earth moon system

int main(){
  constexpr int num_double_in_SIMD_register = 4;  //avx2 -> 256bit -> 4 doubles

  constexpr int num_big_steps = 100; //quantifies samples in outputfile
  constexpr double t_delta = 5*(60*60*24);
  constexpr double steps_per_second = 10; //0.1-10 seems decent
  constexpr int64_t num_total_steps = ceil((steps_per_second * t_delta)
                                            /num_big_steps)*num_big_steps;
  constexpr int num_substeps_per_big_step = num_total_steps / num_big_steps;
  constexpr double stepsize = t_delta / num_total_steps;

  constexpr int num_particles = 100;
  constexpr double v_deviation_sigma = 0;
  constexpr double disc_thickness_sigma = 50;
  constexpr double ring_radius = 1.12e+8;
  constexpr double ring_width_sigma = 3.0e+7;
  constexpr double mass_sigma = 0.1;

  //specify the num of threads that should be used
  //basicaly no overhead for == 1
  int num_threads = std::thread::hardware_concurrency();
  //int num_threads = 1;

  ParticleSimulation my_sim(num_particles, num_big_steps,
                            num_substeps_per_big_step, stepsize, num_threads);

  //std::random_device r;
  //std::default_random_engine engine(r());
  std::default_random_engine engine(4);
  std::normal_distribution<double> disc_height_distro(0, disc_thickness_sigma);
  std::normal_distribution<double> disc_radius_distro(0, ring_width_sigma);
  std::uniform_real_distribution<double> phase_distro(0, 2*M_PI);
  std::normal_distribution<double> v_dev_distro(0, v_deviation_sigma);
  std::normal_distribution<double> mass_distro(0, mass_sigma);

  InPutParticle<double> input;
  input = BodyAtCenter(5.683e+26);  //saturn
  my_sim.SetParticle(input.particle, input.m);

  //ringparticles
  for(int i=1; i<num_particles; i++){
    double x_deviation[3] = {disc_radius_distro(engine),
                             disc_height_distro(engine),0};
    double y_deviation[3] = {v_dev_distro(engine), v_dev_distro(engine),
                             v_dev_distro(engine)};
    input = NonCircularOrbitMoon(x_deviation, y_deviation, phase_distro(engine),
                                 0.0, 0.0, true, 5.683e+26,
                                 pow(mass_distro(engine), 2), ring_radius);
    my_sim.SetParticle(input.particle, input.m);
  }

  my_sim.SetMissingParticlesInBatch();

  //choose sim option
  my_sim.SimulationCPU();
  //my_sim.SimulationAVX2();
  //my_sim.SimulationGPU();
  //my_sim.SimulationGPUCPU();

  my_sim.WriteParticleFiles("../data/raw/earth_mmon/");
  my_sim.WriteTimestepFiles("../data/raw/earth_mmon/");

  my_sim.PrintAverageStepTime();
}
