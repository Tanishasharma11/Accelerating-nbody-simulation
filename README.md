# Accelerating-nbody-simulation

An n-body simulator predicts the individual motions of a group of objects interacting with each other gravitationally. 01-nbody.cu contains a simple, though working, n-body simulator for bodies moving through 3 dimensional space.
In its current CPU-only form, this application takes about 5 seconds to run on 4096 particles, and 20 minutes to run on 65536 particles. We aim to GPU accelerate the program, retaining the correctness of the simulation.
