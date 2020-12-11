#pragma once

#include "immintrin.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <thread>
//Condition_variable solution used instead of latch since latch is not
//yet implemented in gcc or clang
#include <mutex>
#include <condition_variable>
///#include <latch>
#include <sstream>
#include <iomanip>

#include "timer.h"

//Includes mainly the class ParticleSimulation, which holds the particles data
//and performs the Simulation

template<typename T>
struct Particle{
  T r[3];
  T v[3];
};

template<typename T>
struct OutPutParticle{
  T r[3];
  T v[3];
  T a[3];
  T m;
  T t;
};

//The batched structs are arranged differently and specificaly alligned to allow
//the usage of aligned avx load instructions
template<typename T, int SIMD_float_t_width>
struct alignas(32) ParticleBatch{
  T r_x[SIMD_float_t_width];
  T r_y[SIMD_float_t_width];
  T r_z[SIMD_float_t_width];
  T v_x[SIMD_float_t_width];
  T v_y[SIMD_float_t_width];
  T v_z[SIMD_float_t_width];
};

template<typename T, int SIMD_float_t_width>
struct alignas(32) ParticleMassBatch{
  T m[SIMD_float_t_width];
};

//Holds the neccesary parameters and data of the particles and performs and
//performs the gravitational N-Body simulation.
//How to use: 1.Instantiate
//            2.Set particle data of all particles with SetParticle()
//            3.If the number of particles % number of particles per batch != 0
//              call SetMissingParticlesInBatch() to correctly initialize the
//              non existing particles (only neccessary if SimulationAVX2() is
//              used)
//            4.Call either of SimulationCPU(), SimulationAVX2, SimulationGPU or
//              SimulationGPUCPU() to perform the simulation
//            5.Write results to file with WriteParticleFiles() or
//              WriteTimestepFiles()
//            6.Get basic stats of the execution time with PrintAverageStepTime
//For multithreading either threads always have to be started at MakeSmallStep..
//or two synchronizations with e.g. two latches have to be used. std::latch isnt
//in gcc or clang yet and custom version was way slower than just starting new
//threads. However this is still a very large overhead. Due to this only use
//multithreading for large N.
template<typename float_t, int SIMD_float_t_width>
class ParticleSimulation{
public:
  ParticleSimulation(int num_particles, int64_t num_big_steps,
                     int64_t num_substeps_per_big_step,
                     float_t stepsize, int num_threads)
      : num_particles_(num_particles),
        num_big_steps_(num_big_steps),
        num_substeps_per_big_step_(num_substeps_per_big_step),
        stepsize_(stepsize),
        num_threads_(num_threads),
        current_big_step_(0),
        particle_set_counter_(0),
        total_execution_time_(0) {
    //Last batch is not neccessarily full
    num_batches_ = std::ceil(static_cast<float>(num_particles_) /
                                                SIMD_float_t_width);
    //For the non avx simulation particles are evenly distributed amoung threads
    //remaing particles are then distributed with increasing thread id
    //e.g. 5 particles, 3 threads -> 2,2,1
    for(int i=0; i<num_threads_; i++){
      num_particles_in_this_thread_.push_back(num_particles_ / num_threads_);
    }
    for(int i=0; i<(num_particles_ % num_threads_); i++){
      num_particles_in_this_thread_[i] ++;
    }
    for(int i=0; i<num_threads_; i++){
      num_particles_before_this_thread_.push_back(0);
      for(int j=0; j<i; j++){
        num_particles_before_this_thread_[i] +=
            num_particles_in_this_thread_[j];
      }
    }
    //Analogous as above for AVX2 version with batches
    for(int i=0; i<num_threads_; i++){
      num_batches_in_this_thread_.push_back(num_batches_ / num_threads_);
    }
    for(int i=0; i<(num_batches_ % num_threads_); i++){
      num_batches_in_this_thread_[i] ++;
    }
    for(int i=0; i<num_threads_; i++){
      num_batches_before_this_thread_.push_back(0);
      for(int j=0; j<i; j++){
        num_batches_before_this_thread_[i] += num_batches_in_this_thread_[j];
      }
    }
    particles_current_step_ =
        std::make_unique<Particle<float_t>[]>(num_particles_);
    particles_next_step_ =
        std::make_unique<Particle<float_t>[]>(num_particles_);
    particle_mass_ = std::make_unique<float_t[]>(num_particles_);
    batched_particles_current_step_ = std::make_unique<ParticleBatch<float_t,
        SIMD_float_t_width>[]>(num_batches_);
    batched_particles_next_step_ = std::make_unique<ParticleBatch<float_t,
        SIMD_float_t_width>[]>(num_batches_);
    particle_batch_masses_ = std::make_unique<ParticleMassBatch<float_t,
        SIMD_float_t_width>[]>(num_batches_);
    out_put_data_.resize(num_big_steps_ + 1,
        std::vector<OutPutParticle<float_t>>(num_particles_));
    execution_time_ = std::make_unique<double[]>(num_big_steps_);
  }

  //Used to initialize data of particles before the simulation
  void SetParticle(Particle<float_t> particle, float_t mass){
    //Set particle data in linearly aranged arrays
    int batch_id = particle_set_counter_ / SIMD_float_t_width;
    int inter_batch_id = particle_set_counter_ % SIMD_float_t_width;
    for(int i=0; i<3; i++){
      particles_current_step_[particle_set_counter_].r[i] = particle.r[i];
      particles_current_step_[particle_set_counter_].v[i] = particle.v[i];
    }
    particle_mass_[particle_set_counter_] = mass;

    //Set particle data in batched arrays
    batched_particles_current_step_[batch_id].r_x[inter_batch_id] =
        particle.r[0];
    batched_particles_current_step_[batch_id].r_y[inter_batch_id] =
        particle.r[1];
    batched_particles_current_step_[batch_id].r_z[inter_batch_id] =
        particle.r[2];
    batched_particles_current_step_[batch_id].v_x[inter_batch_id] =
        particle.v[0];
    batched_particles_current_step_[batch_id].v_y[inter_batch_id] =
        particle.v[1];
    batched_particles_current_step_[batch_id].v_z[inter_batch_id] =
        particle.v[2];
    particle_batch_masses_[batch_id].m[inter_batch_id] = mass;

    particle_set_counter_ ++;
  }

  //Used to initialize the data of "nonexisting" particles in last batch
  //This is needed for funtional correctness of simulation, if last batch isnt
  //full and SimulateAVX2() is used!
  void SetMissingParticlesInBatch(){
    if (particle_set_counter_ < num_particles_) {
      for(int i=(particle_set_counter_ % SIMD_float_t_width);
          i<SIMD_float_t_width; i++){
            batched_particles_current_step_[num_batches_ - 1].r_x[i] = 0;
            batched_particles_current_step_[num_batches_ - 1].r_y[i] = 0;
            batched_particles_current_step_[num_batches_ - 1].r_z[i] = 0;
            batched_particles_current_step_[num_batches_ - 1].v_x[i] = 0;
            batched_particles_current_step_[num_batches_ - 1].v_y[i] = 0;
            batched_particles_current_step_[num_batches_ - 1].v_z[i] = 0;
            particle_batch_masses_[num_batches_ - 1].m[i] = 0;
      }
    }
  }

  //Executes the simulation on the set particles, given the parameters supplied
  //during instantiation.
  //Is multithreaded for num_particles_ > 1
  void SimulateCPU(){
    IntervallTimer sim_timer;
    //Save initial particle data
    SaveFirstStep();

    for(int i=0; i<num_big_steps_; i++){
      IntervallTimer big_step_timer;
      current_big_step_++;
      MakeBigStep();
      std::cout << (static_cast<float_t>(current_big_step_) /
                    num_big_steps_) * 100
                << "% done" << std::endl;
      execution_time_[i] =
          static_cast<double>(big_step_timer.getTimeInMicroseconds()) * 1.0e-06;
    }

    total_execution_time_ = sim_timer.getTimeInSeconds();
  }

  void MakeBigStep(){
    for(int i=0; i<(num_substeps_per_big_step_-1); i++){
      MakeSmallStepAllThreads(false);
    }
    MakeSmallStepAllThreads(true);
  }

  //Handles the multithreaded execution of MakeSmallStepAVX2()
  void MakeSmallStepAllThreads(bool is_last_small_step){
    std::vector<std::thread> worker_threads;
    for(int i=0; i<(num_threads_-1); i++){
      worker_threads.push_back(std::thread(&ParticleSimulation::MakeSmallStep,
                               this, i + 1, is_last_small_step));
    }
    MakeSmallStep(0, is_last_small_step);

    for(int i=0; i<(num_threads_-1); i++){
      worker_threads[i].join();
    }

    //The calculated new data, that was temporarily stored in
    //particles_next_step_, is now put into particles_current_step_ to act as
    //the input data for the next step. Since the particles_current_step_ is
    //read only and the particles_next_step_ is write only during the
    //calculation this can be done by just switching the pointers.
    particles_current_step_.swap(particles_next_step_);
  }

  //Computes the acceleration on each particle that this thread is responsible
  //for and computes from that the velocity and postion in the next step.
  //The bool is_last_small_step makes sure the acceleration data is exported
  //for the last small step within a big step.
  //Uses AVX2 instructions -> Executing CPU must support these (alternative is
  //SimulateCPU() or SimulateGPU())
  void MakeSmallStep(int thread_id, bool is_last_small_step){
    //Loop over the particles that "belong" to this thread
    for(int i=num_particles_before_this_thread_[thread_id];
        i<num_particles_before_this_thread_[thread_id] +
            num_particles_in_this_thread_[thread_id];
        i++){
      float_t a[3] = {0,0,0};
      //Computes and adds together all accelerations due to other bodies
      for(int j=0; j<num_particles_; j++){
        float_t delta_x = particles_current_step_[j].r[0] -
            particles_current_step_[i].r[0];
        float_t delta_y = particles_current_step_[j].r[1] -
            particles_current_step_[i].r[1];
        float_t delta_z = particles_current_step_[j].r[2] -
            particles_current_step_[i].r[2];
        float_t r_squared = (delta_x * delta_x) +
                            (delta_y * delta_y) +
                            (delta_z * delta_z);
        //Force diverges for r=0
        //Deals with the "unlucky" cases where particles end up in the same
        //place by chances, as well as with the i=j case.
        if (r_squared != 0) {
          float_t tmp = particle_mass_[j] / (r_squared*sqrt(r_squared));
          a[0] += (tmp * delta_x);
          a[1] += (tmp * delta_y);
          a[2] += (tmp * delta_z);
        }
      }

      //Additional forces could be added here e.g. from a position depended
      //forcefield

      //Computes and stores changes in position and velocity in
      //particles_next_step_.
      //Next positions and velocities are calculated via an inline Euler.
      //Higher order methods like runge kutta are not used since they call
      //f(y,t) additional times at different y,t to increase accuracy, but since
      //f(y,t) = acceleration[i] (which is const, i.e. acceleration does not
      //depend on velocity or time) -> higher order methods are more expensive
      //versions of euler in our case.
      particles_next_step_[i].v[0] = particles_current_step_[i].v[0] +
                                          (stepsize_ * a[0] * G_);
      particles_next_step_[i].v[1] = particles_current_step_[i].v[1] +
                                          (stepsize_ * a[1] * G_);
      particles_next_step_[i].v[2] = particles_current_step_[i].v[2] +
                                          (stepsize_ * a[2] * G_);

      particles_next_step_[i].r[0] = particles_current_step_[i].r[0] +
          (stepsize_ * particles_current_step_[i].v[0]);
      particles_next_step_[i].r[1] = particles_current_step_[i].r[1] +
          (stepsize_ * particles_current_step_[i].v[1]);
      particles_next_step_[i].r[2] = particles_current_step_[i].r[2] +
          (stepsize_ * particles_current_step_[i].v[2]);

      //Save the output data of the particle in out_put_data_
      //if it is the last small step and discard it otherwise.
      if (is_last_small_step) {
        out_put_data_[current_big_step_][i].r[0] = particles_next_step_[i].r[0];
        out_put_data_[current_big_step_][i].r[1] = particles_next_step_[i].r[1];
        out_put_data_[current_big_step_][i].r[2] = particles_next_step_[i].r[2];

        out_put_data_[current_big_step_][i].v[0] = particles_next_step_[i].v[0];
        out_put_data_[current_big_step_][i].v[1] = particles_next_step_[i].v[1];
        out_put_data_[current_big_step_][i].v[2] = particles_next_step_[i].v[2];

        out_put_data_[current_big_step_][i].a[0] = a[0] * G_;
        out_put_data_[current_big_step_][i].a[1] = a[1] * G_;
        out_put_data_[current_big_step_][i].a[2] = a[2] * G_;

        out_put_data_[current_big_step_][i].m = particle_mass_[i];
        out_put_data_[current_big_step_][i].t = stepsize_ * current_big_step_ *
                                                num_substeps_per_big_step_;
      }
    }
  }

  //Executes the simulation on the set particles, given the parameters supplied
  //during instantiation.
  //Is multithreaded for num_particles_ > 1
  //Uses AVX2 instructions -> Executing CPU must support these (alternative is
  //SimulateCPU() or SimulateGPU())
  void SimulateAVX2(){
    IntervallTimer sim_timer;
    //Initial particle data is saved
    SaveFirstStep();

    for(int i=0; i<num_big_steps_; i++){
      IntervallTimer big_step_timer;
      current_big_step_++;
      MakeBigStepAVX2();
      std::cout << (static_cast<float_t>(current_big_step_) /
                                        num_big_steps_)*100
                << "% done" << std::endl;
      execution_time_[i] =
          static_cast<double>(big_step_timer.getTimeInMicroseconds()) * 1.0e-06;
    }

    total_execution_time_ = sim_timer.getTimeInSeconds();
  }


  void MakeBigStepAVX2(){
    for(int i=0; i<(num_substeps_per_big_step_-1); i++){
      MakeSmallStepAllThreadsAVX2(false);
    }
    MakeSmallStepAllThreadsAVX2(true);
  }

  //Handles the multithreaded execution of MakeSmallStepAVX2()
  void MakeSmallStepAllThreadsAVX2(bool is_last_small_step){
    std::vector<std::thread> worker_threads;
    for(int i=0; i<(num_threads_-1); i++){
      worker_threads.push_back(
          std::thread(&ParticleSimulation::MakeSmallStepAVX2, this, i + 1,
                      is_last_small_step));
    }
    MakeSmallStepAVX2(0, is_last_small_step);

    for(int i=0; i<(num_threads_-1); i++){
      worker_threads[i].join();
    }

    //The calculated new data that was temporarily stored in
    //particles_next_step_ is now put into particles_current_step_ to act as the
    //input data for the next step. Since the particles_current_step_ is read
    //only and the particles_next_step_ is write only during the calculation
    //this can be done by just switching the pointers.
    batched_particles_current_step_.swap(batched_particles_next_step_);
  }

  //Computes the acceleration on each particle that this thread is responsible
  //for and computes from that the velocity and postion in the next step.
  //The bool is_last_small_step makes sure the acceleration data is exported
  //for the last small step within a big step.
  //Uses AVX2 instructions -> Executing CPU must support these (alternative is
  //SimulateCPU() or SimulateGPU())
  void MakeSmallStepAVX2(int thread_id, bool is_last_small_step){
    //Loop over batches that "belong" to this thread
    for(int batch_id=num_batches_before_this_thread_[thread_id];
        batch_id<(num_batches_before_this_thread_[thread_id] +
            num_batches_in_this_thread_[thread_id]);
        batch_id++){
      //Loop over particles in batch
      for(int inter_batch_id=0; inter_batch_id<SIMD_float_t_width;
          inter_batch_id++){
        //Set one vector each, filled with x,y,z position of current particle.
        __m256d current_x = _mm256_set1_pd(
            batched_particles_current_step_[batch_id].r_x[inter_batch_id]);
        __m256d current_y = _mm256_set1_pd(
            batched_particles_current_step_[batch_id].r_y[inter_batch_id]);
        __m256d current_z = _mm256_set1_pd(
            batched_particles_current_step_[batch_id].r_z[inter_batch_id]);
        __m256d acceleration_x = _mm256_setzero_pd();
        __m256d acceleration_y = _mm256_setzero_pd();
        __m256d acceleration_z = _mm256_setzero_pd();
        //Vector filled with 0 for later comparision
        __m256d zeros = _mm256_setzero_pd();

        //Compute acceleration for one particle
        for(int i=0; i<num_batches_; i++){
          //Fetch one batch worth off other particle positions and massses
          //stored via one component in each vector register.
          __m256d other_x = _mm256_load_pd(
              batched_particles_current_step_[i].r_x);
          __m256d other_y = _mm256_load_pd(
              batched_particles_current_step_[i].r_y);
          __m256d other_z = _mm256_load_pd(
              batched_particles_current_step_[i].r_z);
          __m256d other_mass = _mm256_load_pd(particle_batch_masses_[i].m);

          //Get componentwise differences
          __m256d delta_x = _mm256_sub_pd (other_x, current_x);
          __m256d delta_y = _mm256_sub_pd (other_y, current_y);
          __m256d delta_z = _mm256_sub_pd (other_z, current_z);

          //Compute r*r from the differences utilizing fused multiply add
          //instructions, that are faster and more precise
          __m256d r_squared = _mm256_fmadd_pd (delta_z, delta_z,
              _mm256_fmadd_pd (delta_y, delta_y,
                               _mm256_mul_pd(delta_x, delta_x)));

          __m256d mask = _mm256_cmp_pd(r_squared, zeros, _CMP_GT_OQ);
          //Check if no delta_r=0 between particles in batch
          if (_mm256_movemask_pd(mask) == 0xf) {
            //TODO: consider rsqrt intrinsic for AVX512 version (avx2 version
            //only availabe for float)
            __m256d tmp = _mm256_div_pd(other_mass, _mm256_mul_pd(r_squared,
                _mm256_sqrt_pd(r_squared)));
            acceleration_x = _mm256_fmadd_pd(tmp, delta_x, acceleration_x);
            acceleration_y = _mm256_fmadd_pd(tmp, delta_y, acceleration_y);
            acceleration_z = _mm256_fmadd_pd(tmp, delta_z, acceleration_z);
          } else {
            //Pardon the c-style casts :)
            //Masked instructions only available in avx512, thus manualy acess
            //the entries in vector registers and do the calculations if r!=0.
            for(int j=0; j<SIMD_float_t_width; j++){
              //If (basicaly r_squared[j] !=0) -> perform calculation
              //Do nothing otherwise.
              if( ((double*)(&r_squared))[j] != 0) {
                double tmp1 = ((double*)(&other_mass))[j] /
                    (((double*)(&r_squared))[j] *
                      sqrt(((double*)(&r_squared))[j]));

                //i.e. acceleration_x[j] = tmp[j]*delta_x[j]+acceleration_x[j]
                ((double*)(&acceleration_x))[j] = tmp1 *
                                                ((double*)(&delta_x))[j] +
                                                ((double*)(&acceleration_x))[j];
                ((double*)(&acceleration_y))[j] = tmp1 *
                                                ((double*)(&delta_y))[j] +
                                                ((double*)(&acceleration_y))[j];
                ((double*)(&acceleration_z))[j] = tmp1 *
                                                ((double*)(&delta_z))[j] +
                                                ((double*)(&acceleration_z))[j];
              }
            }
          }
        }

        //Reduce acceleration_i, multiplay by G_ to get the real acceleration
        //and store in final_acceleration
        double final_acceleration[3];
        acceleration_x = _mm256_hadd_pd(acceleration_x, acceleration_x);
        final_acceleration[0] = G_ * (((double*)(&acceleration_x))[0] +
                                ((double*)(&acceleration_x))[2]);
        acceleration_y = _mm256_hadd_pd(acceleration_y, acceleration_y);
        final_acceleration[1] = G_ * (((double*)(&acceleration_y))[0] +
                                ((double*)(&acceleration_y))[2]);
        acceleration_z = _mm256_hadd_pd(acceleration_z, acceleration_z);
        final_acceleration[2] = G_ * (((double*)(&acceleration_z))[0] +
                                ((double*)(&acceleration_z))[2]);

        //Additional forces could be added here e.g. from a position depended
        //forcefield

        //Computes and stores changes in position and velocity in
        //particles_next_step_.
        //Next positions and velocities are calculated via an inline Euler.
        //Higher order methods like runge kutta are not used since they call
        //f(y,t) additional times at different y,t to increase accuracy, but
        //since f(y,t) = acceleration[i] (which is const, i.e. acceleration does
        //not depend on velocity or time) -> higher order methods are more
        //expensive versions of euler in our case.
        batched_particles_next_step_[batch_id].v_x[inter_batch_id] =
            batched_particles_current_step_[batch_id].v_x[inter_batch_id] +
            (stepsize_ * final_acceleration[0]);
        batched_particles_next_step_[batch_id].v_y[inter_batch_id] =
            batched_particles_current_step_[batch_id].v_y[inter_batch_id] +
            (stepsize_ * final_acceleration[1]);
        batched_particles_next_step_[batch_id].v_z[inter_batch_id] =
            batched_particles_current_step_[batch_id].v_z[inter_batch_id] +
            (stepsize_ * final_acceleration[2]);

        batched_particles_next_step_[batch_id].r_x[inter_batch_id] =
            batched_particles_current_step_[batch_id].r_x[inter_batch_id] +
            (stepsize_ *
            batched_particles_current_step_[batch_id].v_x[inter_batch_id]);
        batched_particles_next_step_[batch_id].r_y[inter_batch_id] =
            batched_particles_current_step_[batch_id].r_y[inter_batch_id] +
            (stepsize_ *
            batched_particles_current_step_[batch_id].v_y[inter_batch_id]);
        batched_particles_next_step_[batch_id].r_z[inter_batch_id] =
            batched_particles_current_step_[batch_id].r_z[inter_batch_id] +
            (stepsize_ *
            batched_particles_current_step_[batch_id].v_z[inter_batch_id]);

        if (is_last_small_step) {
          int particle_id = batch_id * SIMD_float_t_width + inter_batch_id;

          out_put_data_[current_big_step_][particle_id].r[0] =
              particles_next_step_[particle_id].r[0];
          out_put_data_[current_big_step_][particle_id].r[1] =
              particles_next_step_[particle_id].r[1];
          out_put_data_[current_big_step_][particle_id].r[2] =
              particles_next_step_[particle_id].r[2];

          out_put_data_[current_big_step_][particle_id].v[0] =
              particles_next_step_[particle_id].v[0];
          out_put_data_[current_big_step_][particle_id].v[1] =
              particles_next_step_[particle_id].v[1];
          out_put_data_[current_big_step_][particle_id].v[2] =
              particles_next_step_[particle_id].v[2];

          out_put_data_[current_big_step_][particle_id].a[0] =
              final_acceleration[0];
          out_put_data_[current_big_step_][particle_id].a[1] =
              final_acceleration[1];
          out_put_data_[current_big_step_][particle_id].a[2] =
              final_acceleration[2];

          out_put_data_[current_big_step_][particle_id].m =
              particle_mass_[particle_id];
          out_put_data_[current_big_step_][particle_id].t =
              stepsize_ * current_big_step_ * num_substeps_per_big_step_;
        }
      }
    }
  }

  //TODO: SimulationGPU()
  //TODO: SimulationGPUCPU() (if it makes senese, which it probably doesnt)

  //Saves initial particle data to out_put_data_
  void SaveFirstStep(){
    for(int i=0; i<num_particles_; i++){
      for(int j=0; j<3; j++){
        out_put_data_[0][i].r[j] = particles_current_step_[i].r[j];
        out_put_data_[0][i].v[j] = particles_current_step_[i].v[j];
        out_put_data_[0][i].a[j] = 0;
      }
      out_put_data_[0][i].m = particle_mass_[i];
      out_put_data_[0][i].t = 0 * num_substeps_per_big_step_ * stepsize_;
    }
  }

  //Writes the data from all big time steps, of one particle, to one file, for
  //each particle.
  void WriteParticleFiles(std::string destination_directory){
    for(int j=0; j<num_particles_; j++){
      std::string file_name = destination_directory;
      file_name += "N_Body_Particle_";
      std::ostringstream ss;
      ss << std::setw(5) << std::setfill('0') << j;
      std::string id = ss.str();
      file_name += id;
      file_name += ".txt";

      std::ofstream my_file(file_name);
      if (my_file.is_open()){
        for(int k=0; k<num_big_steps_+1; k++){
          my_file << (k * num_substeps_per_big_step_ * stepsize_)
                  << "  "
                  << out_put_data_[k][j].r[0] << " "
                  << out_put_data_[k][j].r[1] << " "
                  << out_put_data_[k][j].r[2] << "  "
                  << out_put_data_[k][j].v[0] << " "
                  << out_put_data_[k][j].v[1] << " "
                  << out_put_data_[k][j].v[2] << "  "
                  << out_put_data_[k][j].a[0] << " "
                  << out_put_data_[k][j].a[1] << " "
                  << out_put_data_[k][j].a[2] << "  "
                  << out_put_data_[k][j].m << "\n";
        }
        my_file.close();
      }else{
        std::cout << "Unable to open file \n";
      }
    }
  }

  //Writes the data from all particles, of one big time step, to one file, for
  //each big time step.
  void WriteTimestepFiles(std::string destination_directory){
    for(int j=0; j<num_big_steps_; j++){
      std::string file_name = destination_directory;
      file_name += "N_Body_Timestep_";
      std::ostringstream ss;
      ss << std::setw(5) << std::setfill('0') << j;
      std::string id = ss.str();
      file_name += id;
      file_name += ".txt";

      std::ofstream my_file(file_name);
      if (my_file.is_open()){
        for(int k=0; k<num_particles_; k++){
          my_file << (j * num_substeps_per_big_step_ * stepsize_)
                  << "  "
                  << out_put_data_[j][k].r[0] << " "
                  << out_put_data_[j][k].r[1] << " "
                  << out_put_data_[j][k].r[2] << "  "
                  << out_put_data_[j][k].v[0] << " "
                  << out_put_data_[j][k].v[1] << " "
                  << out_put_data_[j][k].v[2] << "  "
                  << out_put_data_[j][k].a[0] << " "
                  << out_put_data_[j][k].a[1] << " "
                  << out_put_data_[j][k].a[2] << "  "
                  << out_put_data_[j][k].m << "\n";
        }
        my_file.close();
      }else{
        std::cout << "Unable to open file \n";
      }
    }
  }

  //Computes and prints the execution time of the simulation, the average
  //execution time of one big step and its standard deviation.
  void PrintAverageStepTime(){
    int seconds = total_execution_time_ % 60;
    int minutes = total_execution_time_ / 60;
    std::cout << "Total simulation time: " << minutes << " min " << seconds
              << " s" << std::endl;
    double average = 0;
    for(int i=0; i<num_big_steps_; i++){
      average += execution_time_[i];
    }
    average = average / num_big_steps_;

    double sigma = 0;
    for(int i=0; i<num_big_steps_; i++){
      sigma += ((execution_time_[i] - average) *
               (execution_time_[i] - average));
    }
    sigma = sqrt(sigma/(num_big_steps_ - 1));
    std::cout << "Average execution time for a big step: " << average << " +- "
              << sigma << std::endl;
  }

private:
  //Parameters:
  static constexpr float_t G_ = 6.7430e-11; //gravitational constant
  int num_particles_;  //total number of particles in simulation
  //number of steps where the result is saved in out_put_data_
  int64_t num_big_steps_;
  int64_t num_substeps_per_big_step_;  //number of steps between each big step
  float_t stepsize_;
  int num_threads_;
  int64_t current_big_step_;
  int num_batches_;  //determined by num_particles_ and SIMD_float_t_width
  std::vector<int> num_particles_in_this_thread_;
  std::vector<int> num_particles_before_this_thread_;
  std::vector<int> num_batches_in_this_thread_;
  std::vector<int> num_batches_before_this_thread_;
  //keeps track of how many particles are initialized
  int particle_set_counter_;
  //Particle data:
  //Linearly aranged particle data for SimulateCPU()
  std::unique_ptr<Particle<float_t>[]> particles_current_step_;
  std::unique_ptr<Particle<float_t>[]> particles_next_step_;
  std::unique_ptr<float_t[]> particle_mass_;
  //Batched particle data for SimulateAVX2()
  std::unique_ptr<ParticleBatch<float_t, SIMD_float_t_width>[]>
      batched_particles_current_step_;
  std::unique_ptr<ParticleBatch<float_t, SIMD_float_t_width>[]>
      batched_particles_next_step_;
  std::unique_ptr<ParticleMassBatch<float_t,SIMD_float_t_width>[]>
      particle_batch_masses_;
  //Output data:
  std::vector<std::vector<OutPutParticle<float_t>>> out_put_data_;
  //Holds acceleration information collected during each big step temporarily
  std::unique_ptr<double[]> execution_time_;
  int64_t total_execution_time_;
};
