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
iterations = 5
wMax = 0.9
wMin = 0.2
c1 = 2
c2 = 2
vMax = 0.2 * np.array(ub-lb)
vMin = -vMax

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


particles = []

for x in range(0, numberParticles):
    particles.append(Particle(np.random.rand(
        1, vars) * (np.array(ub) - np.array(lb)) + np.array(lb), np.ones(vars), [], np.zeros(vars), float(np.inf)))

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
        particles[i].v = w * particles[i].v + c1 * np.random.rand(1, vars) * (particles[i].pBestX - particles[i].x) + \
            c2 * np.random.rand(1, vars) * \
            (particles[i].pBestX - particles[i].x)

        index1 = np.argwhere(np.ravel(particles[i].v) > vMax)
        index2 = np.argwhere(np.ravel(particles[i].v) < vMin)
        print(index1)
        print(particles[i].v)
        print(particles[i].v[0, index1])

        particles[i].x = particles[i].x + particles[i].v
    print('Iteration #', j, '\n Swarm global best: ', s.gBestO,
          '\n ')
