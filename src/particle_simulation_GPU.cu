#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>

#include "particle_simulation.h"
#include "timer.h"

//Calls cudaMalloc to allocate memory for the needed data arrays on the device
//Returns true if successfull and false for any allocation error
template<typename float_t, int SIMD_float_t_width>
bool ParticleSimulation<float_t, SIMD_float_t_width>::AllocateDeviceMemory(){
  if (cudaMalloc(&dptr_particles_current_step_,
                 sizeof(Particle<float_t>) * num_particles_)
      != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&dptr_particles_next_step_, sizeof(Particle<float_t>) *
                 num_particles_)
      != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&dptr_particle_mass_, sizeof(float_t) * num_particles_)
      != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&dptr_output_data_, sizeof(DeviceOutputParticle<float_t>) *
                 num_particles_ * num_big_steps_)
      != cudaSuccess) {
    return false;
  }
  return true;
}

//Calls cudaMemcpy to copy the particle data to the device.
//Returns true if successfull and false for any error during or before
//transmition
template<typename float_t, int SIMD_float_t_width>
bool ParticleSimulation<float_t, SIMD_float_t_width>::CopyDataToDevice(){
  if (cudaMemcpy(dptr_particles_current_step_, particles_current_step_.get(),
                 sizeof(Particle<float_t>) * num_particles_,
                 cudaMemcpyHostToDevice)
      != cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(dptr_particle_mass_, particle_mass_.get(),
                 sizeof(float_t) * num_particles_, cudaMemcpyHostToDevice)
      != cudaSuccess) {
    return false;
  }
  return true;
}


//TODO: use shared memory for particle data
//Device kernel whoes threads each calculate the change of postion and velocity
//of one particle due to the gravitational interaction with all other particles
template<typename float_t>
__global__ void MakeSmallStepDevice(
      Particle<float_t>* dptr_particles_current_step,
      Particle<float_t>* dptr_particles_next_step,
      float_t* dptr_particle_mass,
      DeviceOutputParticle<float_t>* dptr_output_data,
      int num_particles,
      float_t stepsize, int current_big_step, bool is_last_small_step){
  //Compute particle id from thread id
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  //Check if id corresponds to a vaild particle
  if (id >= num_particles) {
    return;
  }
  float_t a[3] = {0,0,0};
  float_t G = 6.7430e-11;

  //Determine acceleration due to the other particles
  for(int i=0; i<num_particles; i++){
    float_t delta_x = dptr_particles_current_step[i].r[0] -
                      dptr_particles_current_step[id].r[0];
    float_t delta_y = dptr_particles_current_step[i].r[1] -
                      dptr_particles_current_step[id].r[1];
    float_t delta_z = dptr_particles_current_step[i].r[2] -
                      dptr_particles_current_step[id].r[2];

    float_t r_squared = fma(delta_x, delta_x, fma(delta_y, delta_y,
                                                  delta_z * delta_z));

    //Force diverges for r=0
    //Deals with the "unlucky" cases where particles end up in the same
    //place by chances, as well as with the i=j case.
    if (r_squared != 0) {
      float_t tmp = dptr_particle_mass[i] * (1/r_squared) * rsqrt(r_squared);
      a[0] += (tmp * delta_x);
      a[1] += (tmp * delta_y);
      a[2] += (tmp * delta_z);
    }
  }

  //Computes and stores changes in position and velocity in
  //dptr_particles_next_step_.
  //Next positions and velocities are calculated via an inline Euler.
  //Higher order methods like runge kutta are not used since they call
  //f(y,t) additional times at different y,t to increase accuracy, but
  //since f(y,t) = acceleration[i] (which is const, i.e. acceleration does
  //not depend on velocity or time) -> higher order methods are more
  //expensive versions of euler in our case.
  dptr_particles_next_step[id].v[0] =
      dptr_particles_current_step[id].v[0] + (stepsize * a[0] * G);
  dptr_particles_next_step[id].v[1] =
      dptr_particles_current_step[id].v[1] + (stepsize * a[1] * G);
  dptr_particles_next_step[id].v[2] =
      dptr_particles_current_step[id].v[2] + (stepsize * a[2] * G);

  dptr_particles_next_step[id].r[0] =
      dptr_particles_current_step[id].r[0] +
      (stepsize * dptr_particles_current_step[id].v[0]);
  dptr_particles_next_step[id].r[1] =
      dptr_particles_current_step[id].r[1] +
      (stepsize * dptr_particles_current_step[id].v[1]);
  dptr_particles_next_step[id].r[2] =
      dptr_particles_current_step[id].r[2] +
      (stepsize * dptr_particles_current_step[id].v[2]);

  //Saves the result in out_put_data_ if on the last small step of a big
  //step
  if (is_last_small_step) {
    int array_offset = (current_big_step * num_particles) + id;

    dptr_output_data[array_offset].r[0] = dptr_particles_next_step[id].r[0];
    dptr_output_data[array_offset].r[1] = dptr_particles_next_step[id].r[1];
    dptr_output_data[array_offset].r[2] = dptr_particles_next_step[id].r[2];

    dptr_output_data[array_offset].v[0] = dptr_particles_next_step[id].v[0];
    dptr_output_data[array_offset].v[1] = dptr_particles_next_step[id].v[1];
    dptr_output_data[array_offset].v[2] = dptr_particles_next_step[id].v[2];

    dptr_output_data[array_offset].a[0] = a[0] * G;
    dptr_output_data[array_offset].a[1] = a[1] * G;
    dptr_output_data[array_offset].a[2] = a[2] * G;
  }
}

//Analogous to MakeBigStep but on the device
template<typename float_t, int SIMD_float_t_width>
void ParticleSimulation<float_t, SIMD_float_t_width>::MakeBigStepDevice(int blocksize, int current_big_step){
  int gridsize = ceil(static_cast<double>(num_particles_) / blocksize);
  for(int i=0; i<(num_substeps_per_big_step_ - 1); i++){
    MakeSmallStepDevice<<<gridsize, blocksize>>>(
        dptr_particles_current_step_, dptr_particles_next_step_,
        dptr_particle_mass_, dptr_output_data_, num_particles_, stepsize_,
        current_big_step, false);

    //Wait untill the calculation of dptr_particles_next_step_[] is complete
    cudaDeviceSynchronize();

    //Swap data, to use data from dptr_particles_next_step_ as
    //dptr_particles_current_step_ in the next call of MakeSmallStepDevice()
    Particle<float_t>* tmp_dptr;
    tmp_dptr = dptr_particles_next_step_;
    dptr_particles_next_step_ = dptr_particles_current_step_;
    dptr_particles_current_step_ = tmp_dptr;
  }
  MakeSmallStepDevice<<<gridsize, blocksize>>>(
      dptr_particles_current_step_, dptr_particles_next_step_,
      dptr_particle_mass_, dptr_output_data_, num_particles_, stepsize_,
      current_big_step, true);

  //Wait untill the calculation of dptr_particles_next_step_ [] is complete
  cudaDeviceSynchronize();

  //Swap data, to use data from dptr_particles_next_step_ as
  //dptr_particles_current_step_ in the next call of MakeSmallStepDevice()
  Particle<float_t>* tmp_dptr;
  tmp_dptr = dptr_particles_next_step_;
  dptr_particles_next_step_ = dptr_particles_current_step_;
  dptr_particles_current_step_ = tmp_dptr;
}

//Performs the simulation
template<typename float_t, int SIMD_float_t_width>
void ParticleSimulation<float_t, SIMD_float_t_width>::SimulateOnDevice(int blocksize){
  IntervallTimer sim_timer;
  for(int i=0; i<num_big_steps_; i++){
    IntervallTimer big_step_timer;
    MakeBigStepDevice(blocksize, i);

    execution_time_[i] =
        static_cast<double>(big_step_timer.getTimeInMicroseconds()) * 1.0e-06;
    PrintSimulationProgress(i+1);
  }
  total_execution_time_ =  sim_timer.getTimeInSeconds();
}

//Transfers the results back to the host
//Returns true if successfull and false for any error during or before
//transmition
template<typename float_t, int SIMD_float_t_width>
bool ParticleSimulation<float_t, SIMD_float_t_width>::CopyResultsToHost(){
  output_data_from_device_ = new DeviceOutputParticle<float_t>[num_particles_ *
                                                               num_big_steps_];
  if (cudaMemcpy(output_data_from_device_, dptr_output_data_,
                 sizeof(DeviceOutputParticle<float_t>) * num_big_steps_ *
                 num_particles_,
                 cudaMemcpyDeviceToHost)
      != cudaSuccess) {
    return false;
  }
  cudaDeviceSynchronize();
  return true;
}

//Frees the device memory allocated by AllocateDeviceMemory() (smart_ptr are
//not available for device memory)
//Returns true if successfull and false for any error during freeing
template<typename float_t, int SIMD_float_t_width>
bool ParticleSimulation<float_t, SIMD_float_t_width>::FreeDeviceMemory(){
  if (cudaFree(dptr_output_data_) != cudaSuccess) {
    return false;
  }
  if (cudaFree(dptr_particle_mass_) != cudaSuccess) {
    return false;
  }
  if (cudaFree(dptr_particles_next_step_) != cudaSuccess) {
    return false;
  }
  if (cudaFree(dptr_particles_current_step_) != cudaSuccess) {
    return false;
  }
  return true;
}

//To increase the speed of the transfer and reduce memory usage on the device
//on the device DeviceOutputParticle instead of OutPutParticle is used.
//This funtion converts the device result stored in DeviceOutputParticles
//into OutPutParticles, stored in out_put_data_, which is needed for
//WriteParticleFiles() and WriteTimestepFiles().
template<typename float_t, int SIMD_float_t_width>
void ParticleSimulation<float_t, SIMD_float_t_width>::PrepareOutputDataOnHost(){
  //Save first time step from initial data set
  for(int i=0; i<num_particles_; i++){
    for(int j=0; j<3; j++){
      out_put_data_[0][i].r[j] = particles_current_step_[i].r[j];
      out_put_data_[0][i].v[j] = particles_current_step_[i].v[j];
      out_put_data_[0][i].a[j] = 0;
    }
    out_put_data_[0][i].m = particle_mass_[i];
    out_put_data_[0][i].t = 0;
  }
  //Save simulation data from device
  for(int i=1; i<num_big_steps_+1; i++){
    for(int j=0; j<num_particles_; j++){
      for(int k=0; k<3; k++){
        out_put_data_[i][j].r[k] =
            output_data_from_device_[(i-1) * num_particles_ + j].r[k];
        out_put_data_[i][j].v[k] =
            output_data_from_device_[(i-1) * num_particles_ + j].v[k];
        out_put_data_[i][j].a[k] =
            output_data_from_device_[(i-1) * num_particles_ + j].a[k];
      }
      out_put_data_[i][j].m = particle_mass_[j];
      out_put_data_[i][j].t = stepsize_ * num_substeps_per_big_step_ * i;
    }
  }
  delete[] output_data_from_device_;
}

//SimulateGPU() performs the simulation on the device analogous to SimulateCPU
//It returns a string that either indicates the success of the simulation or
//specifies the scource of failure.
//The parameter blocksize can be varried to improve performance. In general
//it should be choosen as a low multiple of the warpsize (32) and the
//resulting amount of blocks (i.e. ceil(num_particles_/blocksize)) should be
//a bit higher than the SM count of the used device (\approx> 2x ?)
//More information is provided in the occupancy section of the cuda
//Best Practises Guide.
template<typename float_t, int SIMD_float_t_width>
std::string ParticleSimulation<float_t, SIMD_float_t_width>::SimulateGPU(int blocksize){
  IntervallTimer device_execution_timer;
  if (!AllocateDeviceMemory()) {
    return "Failed memory allocation on device.";
  }
  if (!CopyDataToDevice()) {
    return "Failed memory copy to device.";
  }

  cudaDeviceSynchronize();
  SimulateOnDevice(blocksize);

  if (!CopyResultsToHost()) {
    return "Failed to copy results to host.";
  }
  if (!FreeDeviceMemory()) {
    return "Failed to free device memory.";
  }

  PrepareOutputDataOnHost();

  std::cout << "Device execution time: "
            << device_execution_timer.getTimeInSeconds() << "s" << std::endl;

  return {"Simulation on GPU succesfull."};
}
