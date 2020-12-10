#include <iostream>
#include <thread>
#include <cstdint>

#include "particle_simulation.h"
#include "particle_initializer.h"

//Example file intended to show how the simulation is set up and performed
//Example is the very unstable and chaotic system of the erth and 7 moons
//that are not all in one plane

int main(){
  constexpr int num_double_in_SIMD_register = 4;  //avx2 -> 256bit -> 4 doubles

  constexpr int num_big_steps = 500; //quantifies samples in outputfile
  constexpr double t_delta = 5*30*(60*60*24);
  constexpr double steps_per_second = 20; //0.1-10 seems decent
  constexpr int64_t num_total_steps = ceil((steps_per_second * t_delta)
                                            /num_big_steps)*num_big_steps;
  constexpr int num_substeps_per_big_step = num_total_steps / num_big_steps;
  constexpr double stepsize = t_delta / num_total_steps;

  //Specify the num of threads that should be used
  //No overhead for == 1
  //Only use multithreading for large N (so not here N=8 :) )
  //int num_threads = std::thread::hardware_concurrency();
  int num_threads = 1;

  //Instantiation doesnt set the particle data
  ParticleSimulation<double,num_double_in_SIMD_register> my_sim(8,
     num_big_steps, num_substeps_per_big_step, stepsize, num_threads);

  //Set particle data
  InPutParticle<double> input;
  input = BodyAtCenter<double>();
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>();
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(M_PI/2, M_PI/2);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(M_PI/2, M_PI/4, M_PI/4);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(2*M_PI/3, M_PI/4, M_PI/4, false);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(7*M_PI/9, 0.0, M_PI/8, false);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(M_PI/5, 0.0, M_PI/3);
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>(M_PI/7, 0.0, M_PI/3);
  my_sim.SetParticle(input.particle, input.m);

  //Is neccesary if N % num_double_in_SIMD_register != 0 (so not needed here)
  my_sim.SetMissingParticlesInBatch();

  //choose sim option
  my_sim.SimulateCPU();
  //my_sim.SimulateAVX2();
  //my_sim.SimulationGPU();
  //my_sim.SimulationGPUCPU();

  my_sim.WriteParticleFiles("");
  //my_sim.WriteTimestepFiles("");

  my_sim.PrintAverageStepTime();
}
