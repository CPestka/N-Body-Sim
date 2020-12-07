#include <iostream>
#include <thread>

#include "particle_simulation.h"
#include "particle_initializer.h"

//example file intended to show how the simulation is set up and performed
//Example is the earth moon system

int main(){
  constexpr int num_double_in_SIMD_register = 4;  //avx2 -> 256bit -> 4 doubles

  constexpr int num_big_steps = 100; //quantifies samples in outputfile
  constexpr double t_delta = 2*30*(60*60*24);
  constexpr double steps_per_second = 10; //0.1-10 seems decent
  constexpr int64_t num_total_steps = ceil((steps_per_second * t_delta)
                                            /num_big_steps)*num_big_steps;
  constexpr int num_substeps_per_big_step = num_total_steps / num_big_steps;
  constexpr double stepsize = t_delta / num_total_steps;

  //specify the num of threads that should be used
  //basicaly no overhead for == 1
  int num_threads = std::thread::hardware_concurrency();
  //int num_threads = 1;

  ParticleSimulation my_sim(8, num_big_steps, num_substeps_per_big_step,
                            stepsize, num_threads);

  InPutParticle<double> input;
  input = BodyAtCenter();
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon();
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI/2, M_PI/2);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI/2, M_PI/4);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI/2, M_PI/4, M_PI/4, false);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI, M_PI/2, M_PI/8, false);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI, 0.0, M_PI/3);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon(M_PI/7, 0.0, M_PI/3);
  my_sim.SetParticle(input.particle, input.m);

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
