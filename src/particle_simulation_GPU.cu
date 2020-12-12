#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <optional>
#include <iostream>

#include "particle_simulation.h"
#include "timer.h"

bool ParticleSimulation::AllocateDeviceMemory(){
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

//Allocates memory for and copies all neccessary data from host to the device
//Returns true if all allocations and memcopys were succesfull and false
//otherwise
bool ParticleSimulation::CopyDataToDevice(){
  if (cudaMemcpy(dptr_particles_current_step_, particles_current_step_,
                 sizeof(Particle<float_t>) * num_particles_),
                 cudaMemcpyHostToDevice)
      != cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(dptr_particle_mass_, particle_mass_,
                 sizeof(float_t) * num_particles_), cudaMemcpyHostToDevice)
      != cudaSuccess) {
    return false;
  }
  return true;
}

void ParticleSimulation::SimulateOnDevice(int blocksize){
  IntervallTimer sim_timer;
  for(int i=0; i<num_big_steps_; i++){
    IntervallTimer big_step_timer;

    MakeBigStepDevice(blocksize, i);

    std::cout << (static_cast<float_t>(i) / num_big_steps_)*100
              << "% done" << std::endl;

    execution_time_[i] =
        static_cast<double>(big_step_timer.getTimeInMicroseconds()) * 1.0e-06;
  }
  total_execution_time_ =  sim_timer.getTimeInSeconds()
}

void ParticleSimulation::MakeBigStepDevice(int blocksize, int current_big_step){
  int gridsize = ceil(static_cast<double>(num_particles_) / blocksize);

  for(int i=0; i<(num_substeps_per_big_step_ - 1); i++){
    MakeSmallStepDevice<<<gridsize, blocksize>>>(
        dptr_particles_current_step_, dptr_particles_next_step_,
        dptr_particle_mass_, dptr_output_data_, num_particles_, stepsize_,
        current_big_step, false);
    //Wait for the calculation of dptr_particles_next_step_ is complete
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
      current_big_step true);
  //Wait for the calculation of dptr_particles_next_step_ is complete
  cudaDeviceSynchronize();
  //Swap data, to use data from dptr_particles_next_step_ as
  //dptr_particles_current_step_ in the next call of MakeSmallStepDevice()
  Particle<float_t>* tmp_dptr;
  tmp_dptr = dptr_particles_next_step_;
  dptr_particles_next_step_ = dptr_particles_current_step_;
  dptr_particles_current_step_ = tmp_dptr;
}

//TODO: use shared memory for particle data
__global__ void MakeSmallStepDevice(
      Particle<float_t>* dptr_particles_current_step,
      Particle<float_t>* dptr_particles_next_step,
      float_t* dptr_particle_mass,
      DeviceOutputParticle<float_t>* dptr_output_data,
      int num_particles,
      float_t stepsize, int current_big_step, bool is_last_small_step){
  //compute particle id from thread  id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  //check if id corresponds to a vaild particle
  if (id >= num_particles) {
    return;
  }
  float_t a[3] = {0,0,0};
  float_t G = 6.7430e-11;

  for(int i=0; i<num_particles; i++){
    float_t delta_x = dptr_particles_current_step[i].r[0] -
                      dptr_particles_current_step[id].r[0];
    float_t delta_y = dptr_particles_current_step[i].r[1] -
                      dptr_particles_current_step[id].r[1];
    float_t delta_z = dptr_particles_current_step[i].r[2] -
                      dptr_particles_current_step[id].r[2];

    float_t r_squared = fma(delta_x, delta_x, fma(delta_y, delta_y,
                                                  delta_x * delta_x));

    if (r_squared != 0) {
      float_t tmp = dptr_particle_mass[i] * (1/r_squared) * rsqrt(r_squared);
      a[0] += (tmp * delta_x);
      a[1] += (tmp * delta_y);
      a[2] += (tmp * delta_z);
    }
  }

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

  if (is_last_small_step) {
    dptr_output_data[current_big_step*num_particles + id].r[0] =
        dptr_particles_next_step[id].r[0];
    dptr_output_data[current_big_step*num_particles + id].r[1] =
        dptr_particles_next_step[id].r[1];
    dptr_output_data[current_big_step*num_particles + id].r[2] =
        dptr_particles_next_step[id].r[2];

    dptr_output_data[current_big_step*num_particles + id].v[0] =
        dptr_particles_next_step[id].v[0];
    dptr_output_data[current_big_step*num_particles + id].v[1] =
        dptr_particles_next_step[id].v[1];
    dptr_output_data[current_big_step*num_particles + id].v[2] =
        dptr_particles_next_step[id].v[2];

    dptr_output_data[current_big_step*num_particles + id].a[0] =
        a[0] * G;
    dptr_output_data[current_big_step*num_particles + id].a[1] =
        a[1] * G;
    dptr_output_data[current_big_step*num_particles + id].a[2] =
        a[2] * G;
  }
}

bool ParticleSimulation::CopyResultsToHost(){
  if (cudaMemcpy(output_data_from_device_, dptr_output_data_,
                 sizeof(DeviceOutputParticle<float_t>) * num_big_steps_,
                 cudaMemcpyDeviceToHost)
      != cudaSuccess) {
    return false;
  }
  return true;
}

bool ParticleSimulation::FreeDeviceMemory(){
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

void ParticleSimulation::PrepareOutputDataOnHost(){
  //Save first time step from initial data set
  for(int i=0; i<num_particles_; i++){
    for(int j=0; j<3; j++){
      out_put_data_[0][i].r[j] = particles_current_step_[i].r[j];
      out_put_data_[0][i].v[j] = particles_current_step_[i].v[j];
      out_put_data_[0][i].a[j] = 0;
    }
    out_put_data_[0][i].m = particle_mass_[i];
    out_put_data_[0][i].t = 0
  }

  //Save simulation data from device
  for(int i=0; i<num_big_steps_; i++){
    for(int j=0; j<num_particles_; i++){
      for(int k=0; k<3; k++){
        out_put_data_[i][j].r[k] =
            output_data_from_device_[i * num_particles_ + j].r[k];
        out_put_data_[i][j].v[k] =
            output_data_from_device_[i * num_particles_ + j].v[k];
        out_put_data_[i][j].a[k] =
            output_data_from_device_[i * num_particles_ + j].a[k];
      }
      out_put_data_[i][j].m = particle_mass_[j];
      out_put_data_[i][j].t = stepsize_ * num_substeps_per_big_step_ * i;
    }
  }
}

std::optional<std::string> ParticleSimulation::SimulateGPU(int blocksize){
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
            << device_execution_timer.getTimeInSeconds() << std::endl;
  return {};
}
