#include <iostream>
#include <thread>
#include <cstdint>

#include "particle_simulation.h"
#include "particle_initializer.h"

//Example file intended to show how the simulation is set up and performed
//Example is the earth moon system. The orbit arround the centere of mass is
//nicely visable

int main(){
  constexpr int num_double_in_SIMD_register = 4;  //avx2 -> 256bit -> 4 doubles

  constexpr int num_big_steps = 400; //quantifies samples in outputfile
  constexpr double t_delta = 5*30*(60*60*24);
  constexpr double steps_per_second = 10; //0.1-10 seems decent
  constexpr int64_t num_total_steps = ceil((steps_per_second * t_delta)
                                            /num_big_steps)*num_big_steps;
  constexpr int num_substeps_per_big_step = num_total_steps / num_big_steps;
  constexpr double stepsize = t_delta / num_total_steps;

  //Specify the num of threads that should be used
  //No overhead for == 1
  //Only use multithreading for large N (so not here N=2 :) )
  //int num_threads = std::thread::hardware_concurrency();
  int num_threads = 1;

  //Instantiation doesnt set the particle data
  ParticleSimulation<double,num_double_in_SIMD_register> my_sim(2,
      num_big_steps, num_substeps_per_big_step, stepsize, num_threads);

  //Set particle data
  InPutParticle<double> input;
  input = BodyAtCenter<double>();
  my_sim.SetParticle(input.particle, input.m);
  input = CircularOrbitMoon<double>();
  my_sim.SetParticle(input.particle, input.m);

  //Is neccesary if N % num_double_in_SIMD_register != 0
  my_sim.SetMissingParticlesInBatch();

  //choose sim option
  my_sim.SimulateCPU();
  //my_sim.SimulateAVX2(); //Doenst make sense for N=2

  my_sim.WriteParticleFiles("");
  //my_sim.WriteTimestepFiles("");

  my_sim.PrintAverageStepTime();
}
