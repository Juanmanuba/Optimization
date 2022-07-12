import numpy as np
import matplotlib.pyplot as plt


def objectiveFunction(x):

    Discrete_set = [10, 20, 30, 40]
    Dx = np.zeros(20)
    idx = 0

    for i in range(0, len(np.ravel(x)), 2):

        if x[0, i] == 0 and x[0, i+1] == 0:
            Dx[idx] = Discrete_set[0]
        elif x[0, i] == 0 and x[0, i+1]:
            Dx[idx] = Discrete_set[1]
        elif x[0, i] and x[0, i+1] == 0:
            Dx[idx] = Discrete_set[2]
        elif x[0, i] and x[0, i+1]:
            Dx[idx] = Discrete_set[3]
        idx = idx + 1
    return np.sum(np.square(Dx))

# Define the details of the discrete optimization problem


bits = 2
vars = 20 * bits
ub = np.ones(vars)
lb = np.zeros(vars)

# Define PSO's parameters

numberParticles = 80
iterations = 50
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
    particles.append(Particle(np.rint(np.random.rand(
        1, vars) * (np.array(ub - lb)) + np.array(lb)), np.ones(vars), [], np.zeros(vars), float(np.inf)))

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
            (s.gBestX - particles[i].x)

        # Check the velocities

        index1 = np.argwhere(np.ravel(particles[i].v) > vMax)
        index2 = np.argwhere(np.ravel(particles[i].v) < vMin)

        particles[i].v[0, index1] = vMax[index1]
        particles[i].v[0, index2] = vMin[index2]

        # sigmoid transfer function

        u = 1 / (1 + np.exp(-particles[i].v))

        # update the position of particle i

        for d in range(0, vars):

            r = np.random.random()

            if r < u[0, d]:
                particles[i].x[0, d] = 1
            else:
                particles[i].x[0, d] = 0

    print('Iteration #', j, '\n Swarm global best: ', s.gBestO,
          '\n ')
