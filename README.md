# N-Body-Sim

Simulation program to compute the movement of N bodies due to their
gravitational interaction.

## Getting started

Clone the repository with

```
git clone https://github.com/corn2424/N-Body-Sim
```

The main functionality is provided by particle_simulation.h.
To use this create your own .cc file which specifies the parameters for the
simulation and the initial positions, velocities and masses of the particles
you which to simulate.
A few toy examples are provided in form of the .cc files and an according .gif,
stored in /data/gif, which shows the result in animated form.

The included examples are:
1. The earth moon system (earth_moon.cc)
2. A hypothetical system containing the earth and 7 moons which don't orbit in one plane, which makes it highly unstable (earth_7moons.cc)
3. A simple example of a slingshoot (aka. swing-by or gravity-assist) with the two bodies being the earth and the moon (slingshoot.cc)
4. A highly simplified version of saturn's ring (saturn_ring.cc)

All .cc files have to be compiled with the flag that allow the usage of AVX2
intrinsics, so e.g. `-march=native` for a machine having native support for AVX2 or `-mavx2` for other machines. This is due to the fact that
SimulateAVX2() makes use of AVX2 intrinsics.
However a alternative not utilizing those is available as well in SimulateCPU()
if support for AVX2 isn't available on a machine.
For multi threading and c++20 feature support the flags `-std_c++2a -pthread` are needed.
Tested with g++ version 10.2.0 .
An example:

```
g++ -std=c++2a -Wall -O3 -mavx2 -march=native -mtune=native -pthread earth_moon.cc -o a.out
```

## Author

Constantin Pestka
