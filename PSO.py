import numpy as np
import matplotlib.pyplot as plt


def objectiveFunction(x):
    return np.sum(np.square(x))

# Define the details of the objective function


vars = 10
ub = 10*np.ones(vars)
lb = -10*np.ones(vars)

# Define PSO's parameters

numberParticles = 30
iterations = 500
wMax = 0.9
wMin = 0.2
c1 = 2
c2 = 2
vMax = 0

# PSO algorithm

# initialize particles


class Swarm:
    def __init__(self, gBestX, gBestO):
        self.gBestX = gBestX
        self.gBestO = gBestO


class Particle:
    def __init__(self, x, v, o, pBestX, pBestO):
        self.x = x
        self.v = v
        self.o = o
        self.pBestX = pBestX
        self.pBestO = pBestO

    def myfunc(self):
        print("Hello my name is " + self.x)


particles = []

for x in range(0, numberParticles):
    p = Particle(np.random.rand(
        1, vars) * (np.array(ub) - np.array(lb)) + np.array(lb), np.zeros(vars), objectiveFunction(0), np.zeros(vars), float(np.inf))
    particles.append(p)

s = Swarm(np.zeros(vars), float(np.inf))

for j in range(0, iterations):

    for i in range(0, numberParticles):  # Calculate the objective value
        currentX = particles[i].x
        particles[i].o = objectiveFunction(currentX)

        if particles[i].o < particles[i].pBestO:  # Update PBest
            particles[i].pBestX = currentX
            particles[i].pBestO = particles[i].o
        if particles[i].o < s.gBestO:  # Update GBest
            s.gBestX = currentX
            s.gBestO = particles[i].o

    # Update x and v for particles
    w = wMax - j * ((wMax - wMin) / iterations)

    for i in range(0, numberParticles):
        particles[i].v = w * 2
