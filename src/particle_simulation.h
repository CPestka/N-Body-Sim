#pragma once

#include "immintrin.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <thread>
//mutex and condition_variable solution used instead of latch since latch is not
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



template<typename float_t, int SIMD_float_t_width>
class ParticleSimulation{
public:
  ParticleSimulation(int num_particles, int64_t num_big_steps,
                     int64_t num_substeps_per_big_step,
                     float_t stepsize, int num_threads)
      : num_particles(num_particles),
        num_big_steps(num_big_steps),
        num_substeps_per_big_step(num_substeps_per_big_step),
        stepsize(stepsize),
        num_threads(num_threads),
        current_big_step(0),
        particle_set_counter(0),
        synch_counter(0) {
    //last batch is not neccessarily full
    this->num_batches = std::ceil(static_cast<float>(this->num_particles)/
                                  SIMD_float_t_width);
    //For the non avx simulation particles are evenly distributed amoung threads
    //remaing particles are then distributed with increasing thread id
    //e.g. 5 particles, 3 threads -> 2,2,1
    for(int i=0; i<this->num_threads; i++){
      this->num_particles_in_this_thread.push_back(this->num_particles /
          this->num_threads);
    }
    for(int i=0; i<(this->num_particles % this->num_threads); i++){
      this->num_particles_in_this_thread[i] ++;
    }
    for(int i=0; i<this->num_threads; i++){
      this->num_particles_before_this_thread.push_back(0);
      for(int j=0; j<i; j++){
        this->num_particles_before_this_thread[i] +=
            this->num_particles_in_this_thread[j];
      }
    }
    //Analogous as above for AVX2 version with batches
    for(int i=0; i<this->num_threads; i++){
      this->num_batches_in_this_thread.push_back(this->num_batches /
          this->num_threads);
    }
    for(int i=0; i<(this->num_batches % this->num_threads); i++){
      this->num_batches_in_this_thread[i] ++;
    }
    for(int i=0; i<this->num_threads; i++){
      this->num_batches_before_this_thread.push_back(0);
      for(int j=0; j<i; j++){
        this->num_batches_before_this_thread[i] +=
            this->num_batches_in_this_thread[j];
      }
    }
    this->particles_current_step = new Particle<float_t>[this->num_particles];
    this->particles_next_step = new Particle<float_t>[this->num_particles];
    this->particle_mass = new float_t[this->num_particles];
    this->batched_particles_current_step =
        new ParticleBatch<float_t, SIMD_float_t_width>[this->num_batches];
    this->batched_particles_next_step =
        new ParticleBatch<float_t, SIMD_float_t_width>[this->num_batches];
    this->particle_batch_masses =
        new ParticleMassBatch<float_t, SIMD_float_t_width>[this->num_batches];
    this->out_put_data.resize(this->num_big_steps + 1,
        std::vector<OutPutParticle<float_t>>(this->num_particles));
    this->a_buffer = new float_t[3*this->num_particles]();
    this->execution_time = new double[this->num_big_steps*this->num_threads];
  }
  ~ParticleSimulation(){
    std::cout << "blah\n";
    delete[] this->execution_time;
    std::cout << "blub\n";
    delete[] this->a_buffer;
    delete[] this->particle_batch_masses;
    delete[] this->batched_particles_next_step;
    delete[] this->batched_particles_current_step;
    delete[] this->particle_mass;
    delete[] this->particles_next_step;
    delete[] this->particles_current_step;
  }

  //Used to initialize data of particles before the simulation
  void SetParticle(Particle<float_t> particle, float_t mass){
    //set particle data in linearly aranged arrays
    int batch_id = this->particle_set_counter / SIMD_float_t_width;
    int inter_batch_id = this->particle_set_counter % SIMD_float_t_width;
    for(int i=0; i<3; i++){
      this->particles_current_step[this->particle_set_counter].r[i] =
          particle.r[i];
      this->particles_current_step[this->particle_set_counter].v[i] =
          particle.v[i];
    }
    this->particle_mass[this->particle_set_counter] = mass;

    //set particle data in batched arrays
    this->batched_particles_current_step[batch_id].r_x[inter_batch_id] =
        particle.r[0];
    this->batched_particles_current_step[batch_id].r_y[inter_batch_id] =
        particle.r[1];
    this->batched_particles_current_step[batch_id].r_z[inter_batch_id] =
        particle.r[2];
    this->batched_particles_current_step[batch_id].v_x[inter_batch_id] =
        particle.v[0];
    this->batched_particles_current_step[batch_id].v_y[inter_batch_id] =
        particle.v[1];
    this->batched_particles_current_step[batch_id].v_z[inter_batch_id] =
        particle.v[2];
    this->particle_batch_masses[batch_id].m[inter_batch_id] = mass;

    this->particle_set_counter ++;
  }

  //Used to initialize the data of "nonexisting" particles in last batch
  //This is needed for funtional correctness of simulation
  void SetMissingParticlesInBatch(){
    if (this->particle_set_counter < this->num_particles) {
      for(int i=(this->particle_set_counter % SIMD_float_t_width);
          i<SIMD_float_t_width; i++){
            this->batched_particles_current_step[this->num_batches - 1].r_x[i] = 0;
            this->batched_particles_current_step[this->num_batches - 1].r_y[i] = 0;
            this->batched_particles_current_step[this->num_batches - 1].r_z[i] = 0;
            this->batched_particles_current_step[this->num_batches - 1].v_x[i] = 0;
            this->batched_particles_current_step[this->num_batches - 1].v_y[i] = 0;
            this->batched_particles_current_step[this->num_batches - 1].v_z[i] = 0;
            this->particle_batch_masses[this->num_batches - 1].m[i] = 0;
      }
    }
  }

  void SimulationCPU(){
    //initial particle data is saved
    SaveCurrentStep();

    std::vector<std::thread> worker_threads;
    for(int i=0; i<(this->num_threads-1); i++){
      worker_threads.push_back(std::thread(&ParticleSimulation::MakeAllSteps,
                               this, i + 1));
    }
    MakeAllSteps(0);

    for(int i=0; i<(this->num_threads-1); i++){
      worker_threads[i].join();
    }
  }

  void MakeAllSteps(int thread_id){
    for(int i=0; i<this->num_big_steps; i++){
      IntervallTimer timer;
      MakeBigStep(thread_id);
      execution_time[(i*thread_id) + thread_id] = timer.getTimeInSeconds();
    }
  }

  void MakeBigStep(int thread_id){
    for(int i=0; i<(this->num_substeps_per_big_step-1); i++){
      MakeSmallStep(thread_id);
    }
    MakeLastSmallStep(thread_id);
    if(thread_id == 0){
      this->current_big_step++;
      SaveCurrentStep();
      std::cout << (static_cast<float_t>(this->current_big_step) /
                                        this->num_big_steps)*100
                << "% done" << std::endl;
    }
  }

  void MakeSmallStep(int thread_id){
    //std::latch step_finished(this->num_threads);

    //over the particles that "belong" to this thread
    for(int i=this->num_particles_before_this_thread[thread_id];
        i<this->num_particles_before_this_thread[thread_id] +
            this->num_particles_in_this_thread[thread_id];
        i++){
      float_t a[3] = {0,0,0};
      //computes and adds together all acceleration due to other bodies
      for(int j=0; j<this->num_particles; j++){
        float_t delta_x = this->particles_current_step[j].r[0] -
            this->particles_current_step[i].r[0];
        float_t delta_y = this->particles_current_step[j].r[1] -
            this->particles_current_step[i].r[1];
        float_t delta_z = this->particles_current_step[j].r[2] -
            this->particles_current_step[i].r[2];
        float_t r_squared = (delta_x * delta_x) +
                            (delta_y * delta_y) +
                            (delta_z * delta_z);
        //Force diverges for r=0
        //deals with "unlucky" cases where particles end up in the same place
        //as well as with the i=j case
        if (r_squared != 0) {
          float_t tmp = particle_mass[j] / (r_squared*sqrt(r_squared));
          a[0] += (tmp * delta_x);
          a[1] += (tmp * delta_y);
          a[2] += (tmp * delta_z);
        }
      }

      //computes and stores changes in position and velocity in
      //particles_next_step
      //Next positions and velocities calculated via an inline Euler
      //higher order methods like runge kutta are not used since they call f(y,t)
      //additional times at different y,t to increase accuracy
      //but since f(y,t) = acceleration[i] (which is const, i.e. acceleration
      //does not depend on velocity or time)
      //-> higher order methods are more expensive versions of euler in our case
      this->particles_next_step[i].v[0] = this->particles_current_step[i].v[0] +
                                          (this->stepsize * a[0] * this->G);
      this->particles_next_step[i].v[1] = this->particles_current_step[i].v[1] +
                                          (this->stepsize * a[1] * this->G);
      this->particles_next_step[i].v[2] = this->particles_current_step[i].v[2] +
                                          (this->stepsize * a[2] * this->G);

      this->particles_next_step[i].r[0] = this->particles_current_step[i].r[0] +
          (this->stepsize * this->particles_current_step[i].v[0]);
      this->particles_next_step[i].r[1] = this->particles_current_step[i].r[1] +
          (this->stepsize * this->particles_current_step[i].v[1]);
      this->particles_next_step[i].r[2] = this->particles_current_step[i].r[2] +
          (this->stepsize * this->particles_current_step[i].v[2]);
    }
    //Synch barrier because all data in particles_next_step musst be calculated
    //before starting the new step
    //step_finished.arrive_and_wait();
    IncrementCounter();
    WaitTillNumThreads();

    //the calculated new data that was temporarily stored in particles_next_step
    //is now put into particles_current_step to act as the input data for the
    //next step. Since the particles_current_step is read only and the
    //particles_next_step is write only during the calculation this can be done
    //by just switching the pointers.
    //std::latch data_ready(this->num_threads);
    if (thread_id == 0) {
      Particle<float_t>* tmp_ptr;
      tmp_ptr = particles_next_step;
      particles_next_step = particles_current_step;
      particles_current_step = tmp_ptr;
    }
    //data_ready.arrive_and_wait();
    DecrementCounter();
    WaitTillZero();
  }

  void MakeLastSmallStep(int thread_id){
    //std::latch step_finished(this->num_threads);

    //over the particles that "belong" to this thread
    for(int i=this->num_particles_before_this_thread[thread_id];
        i<this->num_particles_before_this_thread[thread_id] +
            this->num_particles_in_this_thread[thread_id];
        i++){
      float_t a[3] = {0,0,0};
      //computes and adds together all acceleration due to other bodies
      for(int j=0; j<this->num_particles; j++){
        float_t delta_x = this->particles_current_step[j].r[0] -
            this->particles_current_step[i].r[0];
        float_t delta_y = this->particles_current_step[j].r[1] -
            this->particles_current_step[i].r[1];
        float_t delta_z = this->particles_current_step[j].r[2] -
            this->particles_current_step[i].r[2];
        float_t r_squared = (delta_x * delta_x) +
                            (delta_y * delta_y) +
                            (delta_z * delta_z);
        //Force diverges for r=0
        //deals with "unlucky" cases where particles end up in the same place
        //as well as with the i=j case
        if (r_squared != 0) {
          float_t tmp = particle_mass[j] / (r_squared*sqrt(r_squared));
          a[0] += (tmp * delta_x);
          a[1] += (tmp * delta_y);
          a[2] += (tmp * delta_z);
        }
      }

      //computes and stores changes in position and velocity in
      //particles_next_step
      //Next positions and velocities calculated via an inline Euler
      //higher order methods like runge kutta are not used since they call f(y,t)
      //additional times at different y,t to increase accuracy
      //but since f(y,t) = acceleration[i] (which is const, i.e. acceleration
      //does not depend on velocity or time)
      //-> higher order methods are more expensive versions of euler in our case
      this->particles_next_step[i].v[0] = this->particles_current_step[i].v[0] +
                                          (this->stepsize * a[0] * this->G);
      this->particles_next_step[i].v[1] = this->particles_current_step[i].v[1] +
                                          (this->stepsize * a[1] * this->G);
      this->particles_next_step[i].v[2] = this->particles_current_step[i].v[2] +
                                          (this->stepsize * a[2] * this->G);

      this->particles_next_step[i].r[0] = this->particles_current_step[i].r[0] +
          (this->stepsize * this->particles_current_step[i].v[0]);
      this->particles_next_step[i].r[1] = this->particles_current_step[i].r[1] +
          (this->stepsize * this->particles_current_step[i].v[1]);
      this->particles_next_step[i].r[2] = this->particles_current_step[i].r[2] +
          (this->stepsize * this->particles_current_step[i].v[2]);

      //save the acceleration of the particle in buffer as it is outputdata
      //if it is the last small step and discarded otherwise
      this->a_buffer[3 * i] = a[0] * this->G;
      this->a_buffer[(3 * i) + 1] = a[1] * this->G;
      this->a_buffer[(3 * i) + 2] = a[2] * this->G;
    }

    //Synch barrier because all data in particles_next_step musst be calculated
    //before starting the new step
    //step_finished.arrive_and_wait();
    IncrementCounter();
    WaitTillNumThreads();

    //the calculated new data that was temporarily stored in particles_next_step
    //is now put into particles_current_step to act as the input data for the
    //next step. Since the particles_current_step is read only and the
    //particles_next_step is write only during the calculation this can be done
    //by just switching the pointers.
    //std::latch data_ready(this->num_threads);
    if(thread_id == 0){
      Particle<float_t>* tmp_ptr;
      tmp_ptr = particles_next_step;
      particles_next_step = particles_current_step;
      particles_current_step = tmp_ptr;
    }
    //data_ready.arrive_and_wait();
    DecrementCounter();
    WaitTillZero();
  }

  void SimulationAVX2(){
    //initial particle data is saved
    SaveCurrentStepBatched();

    std::vector<std::thread> worker_threads;
    for(int i=0; i<(this->num_threads-1); i++){
      worker_threads.push_back(std::thread(&ParticleSimulation::MakeAllStepsAVX2,
                               this, i + 1));
    }
    MakeAllStepsAVX2(0);

    for(int i=0; i<(this->num_threads-1); i++){
      worker_threads[i].join();
    }
  }

  void MakeAllStepsAVX2(int thread_id){
    for(int i=0; i<this->num_big_steps; i++){
      IntervallTimer timer;
      MakeBigStepAVX2(thread_id);
      execution_time[(i*thread_id) + thread_id] = timer.getTimeInSeconds();
    }
  }

  void MakeBigStepAVX2(int thread_id){
    for(int i=0; i<(this->num_substeps_per_big_step-1); i++){
      MakeSmallStepAVX2(thread_id);
    }
    MakeLastSmallStepAVX2(thread_id);
    if (thread_id == 0) {
      this->current_big_step++;
      SaveCurrentStepBatched();
      std::cout << (static_cast<float_t>(this->current_big_step) /
                                        this->num_big_steps)*100
                << "% done" << std::endl;
    }
  }

  void MakeSmallStepAVX2(int thread_id){
    //std::latch step_finished(this->num_threads);

    //loop over batches that "belong" to this thread
    for(int batch_id=this->num_batches_before_this_thread[thread_id];
        batch_id<(this->num_batches_before_this_thread[thread_id] +
            this->num_batches_in_this_thread[thread_id]);
        batch_id++){
      //loop over particles in batch
      for(int inter_batch_id=0; inter_batch_id<SIMD_float_t_width;
          inter_batch_id++){
        //set one vector each filled with x,y,z position of current particle
        __m256d current_x = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_x[inter_batch_id]);
        __m256d current_y = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_y[inter_batch_id]);
        __m256d current_z = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_z[inter_batch_id]);
        __m256d acceleration_x = _mm256_setzero_pd();
        __m256d acceleration_y = _mm256_setzero_pd();
        __m256d acceleration_z = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();  //vector filled with 0 for comparision

        //compute acceleration for one particle
        for(int i=0; i<this->num_batches; i++){
          //fetch one batch worth off other particle positions and massses stored
          //via one component in each vector register
          __m256d other_x = _mm256_load_pd(this->batched_particles_current_step[i].r_x);
          __m256d other_y = _mm256_load_pd(this->batched_particles_current_step[i].r_y);
          __m256d other_z = _mm256_load_pd(this->batched_particles_current_step[i].r_z);
          __m256d other_mass = _mm256_load_pd(this->particle_batch_masses[i].m);

          //get componentwise differences
          __m256d delta_x = _mm256_sub_pd (other_x, current_x);
          __m256d delta_y = _mm256_sub_pd (other_y, current_y);
          __m256d delta_z = _mm256_sub_pd (other_z, current_z);

          //compute r*r from the differences utilizing fused multiply add
          //instructions, that are faster and more precise
          __m256d r_squared = _mm256_fmadd_pd (delta_z, delta_z,
              _mm256_fmadd_pd (delta_y, delta_y, _mm256_mul_pd(delta_x, delta_x)));

          __m256d mask = _mm256_cmp_pd(r_squared, zeros, _CMP_GT_OQ);
          //no delta_r=0 between particles in batch
          if (_mm256_movemask_pd(mask) == 0xf) {
            //alternatively use rsqrt (MUCH! faster); only float variant for avx;
            //avx512 has way better rsqrt implementations
            __m256d tmp = _mm256_div_pd(other_mass, _mm256_mul_pd(r_squared,
                      _mm256_sqrt_pd(r_squared)));
            acceleration_x = _mm256_fmadd_pd(tmp, delta_x, acceleration_x);
            acceleration_y = _mm256_fmadd_pd(tmp, delta_y, acceleration_y);
            acceleration_z = _mm256_fmadd_pd(tmp, delta_z, acceleration_z);
          } else {
            //masked instructions not available in non avx512, thus manualy acess
            //the entries in vector registers and do the calculations if r!=0
            for(int j=0; j<SIMD_float_t_width; j++){
              //if (basicaly r_squared[j] !=0) -> perform calculation
              //do nothing otherwise
              if( ((double*)(&r_squared))[j] != 0) {
                double tmp1 = ((double*)(&other_mass))[j] /
                    (((double*)(&r_squared))[j] * sqrt(((double*)(&r_squared))[j]));

                //basically acceleration_x[j] = tmp[j]*delta_x[j]+acceleration_x[j]
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

        //reduce acceleration_i and store in final_acceleration
        double final_acceleration[3];
        acceleration_x = _mm256_hadd_pd(acceleration_x, acceleration_x);
        final_acceleration[0] = this->G * (((double*)(&acceleration_x))[0] +
                                ((double*)(&acceleration_x))[2]);
        acceleration_y = _mm256_hadd_pd(acceleration_y, acceleration_y);
        final_acceleration[1] = this->G * (((double*)(&acceleration_y))[0] +
                                ((double*)(&acceleration_y))[2]);
        acceleration_z = _mm256_hadd_pd(acceleration_z, acceleration_z);
        final_acceleration[2] = this->G * (((double*)(&acceleration_z))[0] +
                                ((double*)(&acceleration_z))[2]);

        //additional forces could be added here e.g. from a position depended forcefield

        //Safe next positions and velocity of current particle
        //
        //Next positions and velocities calculated via an inline Euler
        //higher order methods like runge kutta are not used since they call f(y,t)
        //additional times at different y,t to increase accuracy
        //but since f(y,t) = acceleration[i] (which is const, i.e. acceleration
        //does not depend on velocity or time)
        //-> higher order methods are more expensive versions of euler in our case
        this->batched_particles_next_step[batch_id].v_x[inter_batch_id] =
            this->batched_particles_current_step[batch_id].v_x[inter_batch_id] +
            (this->stepsize * final_acceleration[0]);
        this->batched_particles_next_step[batch_id].v_y[inter_batch_id] =
            this->batched_particles_current_step[batch_id].v_y[inter_batch_id] +
            (this->stepsize * final_acceleration[1]);
        this->batched_particles_next_step[batch_id].v_z[inter_batch_id] =
            this->batched_particles_current_step[batch_id].v_z[inter_batch_id] +
            (this->stepsize * final_acceleration[2]);

        this->batched_particles_next_step[batch_id].r_x[inter_batch_id] =
            this->batched_particles_current_step[batch_id].r_x[inter_batch_id] +
            (this->stepsize *
            this->batched_particles_current_step[batch_id].v_x[inter_batch_id]);
        this->batched_particles_next_step[batch_id].r_y[inter_batch_id] =
            this->batched_particles_current_step[batch_id].r_y[inter_batch_id] +
            (this->stepsize *
            this->batched_particles_current_step[batch_id].v_y[inter_batch_id]);
        this->batched_particles_next_step[batch_id].r_z[inter_batch_id] =
            this->batched_particles_current_step[batch_id].r_z[inter_batch_id] +
            (this->stepsize *
            this->batched_particles_current_step[batch_id].v_z[inter_batch_id]);
      }
    }

    //Synch barrier because all data in particles_next_step musst be calculated
    //before starting the new step
    //step_finished.arrive_and_wait();
    IncrementCounter();
    WaitTillNumThreads();

    //the calculated new data that was temporarily stored in particles_next_step
    //is now put into particles_current_step to act as the input data for the
    //next step. Since the particles_current_step is read only and the
    //particles_next_step is write only during the calculation this can be done
    //by just switching the pointers.
    //std::latch data_ready(this->num_threads);
    if(thread_id == 0){
      ParticleBatch<float_t,SIMD_float_t_width>* tmp_ptr;
      tmp_ptr = batched_particles_next_step;
      batched_particles_next_step = batched_particles_current_step;
      batched_particles_current_step = tmp_ptr;
    }
    //data_ready.arrive_and_wait();
    DecrementCounter();
    WaitTillZero();
  }

  void MakeLastSmallStepAVX2(int thread_id){
    //std::latch step_finished(this->num_threads);

    //loop over batches that "belong" to this thread
    for(int batch_id=this->num_batches_before_this_thread[thread_id];
        batch_id<(this->num_batches_before_this_thread[thread_id] +
            this->num_batches_in_this_thread[thread_id]);
        batch_id++){
      //loop over particles in batch
      for(int inter_batch_id=0; inter_batch_id<SIMD_float_t_width;
          inter_batch_id++){
        //set one vector each filled with x,y,z position of current particle
        __m256d current_x = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_x[inter_batch_id]);
        __m256d current_y = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_y[inter_batch_id]);
        __m256d current_z = _mm256_set1_pd(
            this->batched_particles_current_step[batch_id].r_z[inter_batch_id]);
        __m256d acceleration_x = _mm256_setzero_pd();
        __m256d acceleration_y = _mm256_setzero_pd();
        __m256d acceleration_z = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();  //vector filled with 0 for comparision

        //compute acceleration for one particle
        for(int i=0; i<this->num_batches; i++){
          //fetch one batch worth off other particle positions and massses stored
          //via one component in each vector register
          __m256d other_x = _mm256_load_pd(this->batched_particles_current_step[i].r_x);
          __m256d other_y = _mm256_load_pd(this->batched_particles_current_step[i].r_y);
          __m256d other_z = _mm256_load_pd(this->batched_particles_current_step[i].r_z);
          __m256d other_mass = _mm256_load_pd(this->particle_batch_masses[i].m);

          //get componentwise differences
          __m256d delta_x = _mm256_sub_pd (other_x, current_x);
          __m256d delta_y = _mm256_sub_pd (other_y, current_y);
          __m256d delta_z = _mm256_sub_pd (other_z, current_z);

          //compute r*r from the differences utilizing fused multiply add
          //instructions, that are faster and more precise
          __m256d r_squared = _mm256_fmadd_pd (delta_z, delta_z,
              _mm256_fmadd_pd (delta_y, delta_y, _mm256_mul_pd(delta_x, delta_x)));

          __m256d mask = _mm256_cmp_pd(r_squared, zeros, _CMP_GT_OQ);
          //no delta_r=0 between particles in batch
          if (_mm256_movemask_pd(mask) == 0xf) {
            //alternatively use rsqrt (MUCH! faster); only float variant for avx;
            //avx512 has way better rsqrt implementations
            __m256d tmp = _mm256_div_pd(other_mass, _mm256_mul_pd(r_squared,
                      _mm256_sqrt_pd(r_squared)));
            acceleration_x = _mm256_fmadd_pd(tmp, delta_x, acceleration_x);
            acceleration_y = _mm256_fmadd_pd(tmp, delta_y, acceleration_y);
            acceleration_z = _mm256_fmadd_pd(tmp, delta_z, acceleration_z);
          } else {
            //masked instructions not available in non avx512, thus manualy acess
            //the entries in vector registers and do the calculations if r!=0
            for(int j=0; j<SIMD_float_t_width; j++){
              //if (basicaly r_squared[j] !=0) -> perform calculation
              //do nothing otherwise
              if( ((double*)(&r_squared))[j] != 0) {
                double tmp1 = ((double*)(&other_mass))[j] /
                    (((double*)(&r_squared))[j] * sqrt(((double*)(&r_squared))[j]));

                //basically acceleration_x[j] = tmp[j]*delta_x[j]+acceleration_x[j]
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

        //reduce acceleration_i and store in final_acceleration
        double final_acceleration[3];
        acceleration_x = _mm256_hadd_pd(acceleration_x, acceleration_x);
        final_acceleration[0] = this->G * (((double*)(&acceleration_x))[0] +
                                ((double*)(&acceleration_x))[2]);
        acceleration_y = _mm256_hadd_pd(acceleration_y, acceleration_y);
        final_acceleration[1] = this->G * (((double*)(&acceleration_y))[0] +
                                ((double*)(&acceleration_y))[2]);
        acceleration_z = _mm256_hadd_pd(acceleration_z, acceleration_z);
        final_acceleration[2] = this->G * (((double*)(&acceleration_z))[0] +
                                ((double*)(&acceleration_z))[2]);

        //additional forces could be added here e.g. from a position depended forcefield

        //Safe next positions and velocity of current particle
        //
        //Next positions and velocities calculated via an inline Euler
        //higher order methods like runge kutta are not used since they call f(y,t)
        //additional times at different y,t to increase accuracy
        //but since f(y,t) = acceleration[i] (which is const, i.e. acceleration
        //does not depend on velocity or time)
        //-> higher order methods are more expensive versions of euler in our case
        this->batched_particles_next_step[batch_id].v_x[inter_batch_id] +=
            this->stepsize * final_acceleration[0];
        this->batched_particles_next_step[batch_id].v_y[inter_batch_id] +=
            this->stepsize * final_acceleration[1];
        this->batched_particles_next_step[batch_id].v_z[inter_batch_id] +=
            this->stepsize * final_acceleration[2];

        this->batched_particles_next_step[batch_id].r_x[inter_batch_id] +=
            this->stepsize *
            this->batched_particles_current_step[batch_id].v_x[inter_batch_id];

        this->batched_particles_next_step[batch_id].r_y[inter_batch_id] +=
            this->stepsize *
            this->batched_particles_current_step[batch_id].v_y[inter_batch_id];

        this->batched_particles_next_step[batch_id].r_z[inter_batch_id] +=
            this->stepsize *
            this->batched_particles_current_step[batch_id].v_z[inter_batch_id];
        //save output acceleration in according buffer
        int particle_id = batch_id * SIMD_float_t_width + inter_batch_id;
        this->a_buffer[3*particle_id] = final_acceleration[0];
        this->a_buffer[(3*particle_id)+1] = final_acceleration[1];
        this->a_buffer[(3*particle_id)+2] = final_acceleration[2];
      }
    }

    //Synch barrier because all data in particles_next_step musst be calculated
    //before starting the new step
    //step_finished.arrive_and_wait();
    IncrementCounter();
    WaitTillNumThreads();

    //the calculated new data that was temporarily stored in particles_next_step
    //is now put into particles_current_step to act as the input data for the
    //next step. Since the particles_current_step is read only and the
    //particles_next_step is write only during the calculation this can be done
    //by just switching the pointers.
    //std::latch data_ready(this->num_threads);
    if(thread_id == 0){
      ParticleBatch<float_t,SIMD_float_t_width>* tmp_ptr;
      tmp_ptr = batched_particles_next_step;
      batched_particles_next_step = batched_particles_current_step;
      batched_particles_current_step = tmp_ptr;
    }
    //data_ready.arrive_and_wait();
    DecrementCounter();
    WaitTillZero();
  }

  //void SimulationGPU()
  //void SimulationGPUCPU()

  void SaveCurrentStep(){
    for(int i=0; i<this->num_particles; i++){
      for(int j=0; j<3; j++){
        this->out_put_data[this->current_big_step][i].r[j] =
            this->particles_current_step[i].r[j];
        this->out_put_data[this->current_big_step][i].v[j] =
            this->particles_current_step[i].v[j];
        this->out_put_data[this->current_big_step][i].a[j] =
            this->a_buffer[(3*i) + j];
      }
      this->out_put_data[this->current_big_step][i].m =
          this->particle_mass[i];
      this->out_put_data[this->current_big_step][i].t =
          this->current_big_step * this->num_substeps_per_big_step *
              this->stepsize;
    }
  }

  void SaveCurrentStepBatched(){
    for(int i=0; i<this->num_particles; i++){
      int batch_id = i / SIMD_float_t_width;
      int inter_batch_id = i % SIMD_float_t_width;

      this->out_put_data[this->current_big_step][i].r[0] =
          this->batched_particles_current_step[batch_id].r_x[inter_batch_id];
      this->out_put_data[this->current_big_step][i].r[1] =
          this->batched_particles_current_step[batch_id].r_y[inter_batch_id];
      this->out_put_data[this->current_big_step][i].r[2] =
          this->batched_particles_current_step[batch_id].r_z[inter_batch_id];

      this->out_put_data[this->current_big_step][i].v[0] =
          this->batched_particles_current_step[batch_id].v_x[inter_batch_id];
      this->out_put_data[this->current_big_step][i].v[1] =
          this->batched_particles_current_step[batch_id].v_y[inter_batch_id];
      this->out_put_data[this->current_big_step][i].v[2] =
          this->batched_particles_current_step[batch_id].v_z[inter_batch_id];

      this->out_put_data[this->current_big_step][i].a[0] =
          this->a_buffer[3*i];
      this->out_put_data[this->current_big_step][i].a[1] =
          this->a_buffer[(3*i) + 1];
      this->out_put_data[this->current_big_step][i].a[2] =
          this->a_buffer[(3*i) + 2];

      this->out_put_data[this->current_big_step][i].m =
          this->particle_batch_masses[batch_id].m[inter_batch_id];
      this->out_put_data[this->current_big_step][i].t =
          this->current_big_step * this->num_substeps_per_big_step *
              this->stepsize;
    }
  }

  void WriteParticleFiles(std::string destination_directory){
    for(int j=0; j<this->num_particles; j++){
      std::string file_name = destination_directory;
      file_name += "N_Body_Particle_";
      std::ostringstream ss;
      ss << std::setw(5) << std::setfill('0') << j;
      std::string id = ss.str();
      file_name += id;
      file_name += ".txt";

      std::ofstream my_file(file_name);
      if (my_file.is_open()){
        for(int k=0; k<this->num_big_steps+1; k++){
          my_file << (k * this->num_substeps_per_big_step * this->stepsize)
                  << "  "
                  << this->out_put_data[k][j].r[0] << " "
                  << this->out_put_data[k][j].r[1] << " "
                  << this->out_put_data[k][j].r[2] << "  "
                  << this->out_put_data[k][j].v[0] << " "
                  << this->out_put_data[k][j].v[1] << " "
                  << this->out_put_data[k][j].v[2] << "  "
                  << this->out_put_data[k][j].a[0] << " "
                  << this->out_put_data[k][j].a[1] << " "
                  << this->out_put_data[k][j].a[2] << "  "
                  << this->out_put_data[k][j].m << "\n";
        }
        my_file.close();
      }else{
        std::cout << "Unable to open file \n";
      }
    }
  }

  void WriteTimestepFiles(std::string destination_directory){
    for(int j=0; j<this->num_big_steps; j++){
      std::string file_name = destination_directory;
      file_name += "N_Body_Timestep_";
      std::ostringstream ss;
      ss << std::setw(5) << std::setfill('0') << j;
      std::string id = ss.str();
      file_name += id;
      file_name += ".txt";

      std::ofstream my_file(file_name);
      if (my_file.is_open()){
        for(int k=0; k<this->num_particles; k++){
          my_file << (j * this->num_substeps_per_big_step * this->stepsize)
                  << "  "
                  << this->out_put_data[j][k].r[0] << " "
                  << this->out_put_data[j][k].r[1] << " "
                  << this->out_put_data[j][k].r[2] << "  "
                  << this->out_put_data[j][k].v[0] << " "
                  << this->out_put_data[j][k].v[1] << " "
                  << this->out_put_data[j][k].v[2] << "  "
                  << this->out_put_data[j][k].a[0] << " "
                  << this->out_put_data[j][k].a[1] << " "
                  << this->out_put_data[j][k].a[2] << "  "
                  << this->out_put_data[j][k].m << "\n";
        }
        my_file.close();
      }else{
        std::cout << "Unable to open file \n";
      }
    }
  }

  void PrintAverageStepTime(){
    double average = 0;
    for(int i=0; i<this->num_big_steps*this->num_threads; i++){
      average += execution_time[i];
    }
    average = average / this->num_big_steps;

    double sigma = 0;
    for(int i=0; i<this->num_big_steps*this->num_threads; i++){
      sigma += (execution_time[i] - average) * (execution_time[i] - average);
    }
    sigma = sqrt(sigma/((this->num_big_steps*this->num_threads) -1));
    std::cout << "Average execution time for a big step: " << average << " +- "
              << sigma << std::endl;
  }

  void IncrementCounter(){
    std::lock_guard<std::mutex> synch_counter_guard(synch_counter_mutex);
    this->synch_counter ++;
  }
  void DecrementCounter(){
    std::lock_guard<std::mutex> synch_counter_guard(synch_counter_mutex);
    this->synch_counter --;
  }

  void WaitTillZero(){
    std::unique_lock<std::mutex> wait_lock(this->synch_counter_mutex);
    this->synch_condition_variable.wait(wait_lock, [&]{
                                                 if (this->synch_counter == 0) {
                                                   return true;
                                                 } else {
                                                   return false;
                                                 }});
    this->synch_condition_variable.notify_all();
  }

  void WaitTillNumThreads(){
    std::unique_lock<std::mutex> wait_lock(this->synch_counter_mutex);
    this->synch_condition_variable.wait(wait_lock, [&]{
                                                 if (this->synch_counter ==
                                                         this->num_threads) {
                                                   return true;
                                                 } else {
                                                   return false;
                                                 }});
    this->synch_condition_variable.notify_all();
  }

private:
  static constexpr float_t G = 6.7430e-11; //gravitational constant
  int num_particles;
  int64_t num_big_steps;
  int64_t num_substeps_per_big_step;
  float_t stepsize;
  int num_threads;
  int64_t current_big_step;
  int num_batches;
  std::vector<int> num_particles_in_this_thread;
  std::vector<int> num_particles_before_this_thread; //these two are used for non AVX
  std::vector<int> num_batches_in_this_thread;
  std::vector<int> num_batches_before_this_thread;  //these two are used for AVX
  Particle<float_t>* particles_current_step;
  Particle<float_t>* particles_next_step;
  float_t* particle_mass;
  ParticleBatch<float_t, SIMD_float_t_width>* batched_particles_current_step;
  ParticleBatch<float_t, SIMD_float_t_width>* batched_particles_next_step;
  ParticleMassBatch<float_t,SIMD_float_t_width>* particle_batch_masses;
  std::vector<std::vector<OutPutParticle<float_t>>> out_put_data;
  int particle_set_counter;
  float_t* a_buffer;
  double* execution_time;
  int synch_counter;
  std::mutex synch_counter_mutex;
  std::condition_variable synch_condition_variable;
};
