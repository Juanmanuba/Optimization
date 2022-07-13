import numpy as np
import matplotlib.pyplot as plt


# def objectiveFunction(x):
#     return np.sum(np.square(x))

def objectiveFunction(x):

    # vars
    len = x[0, 0]
    width = x[0, 1]

    # List of your constraints
    const1 = width >= 5 and width <= 6.4
    const2 = (np.square(len) + np.square(width)) >= 4
    const3 = len != width

    if const1 and const2 and const3:
        return 1 * (len + width)

    else:  # penalize the non-feasible solution
        return 1 * (len + width) + 2000

# Define the details of the objective function

# vars = 10
# ub = 10*np.ones(vars)
# lb = -10*np.ones(vars)


vars = 2
ub = 7 * np.ones(vars)
lb = 2 * np.ones(vars)

# Define PSO's parameters

numberParticles = 5
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
            (s.gBestX - particles[i].x)

        # Check the velocities

        index1 = np.argwhere(np.ravel(particles[i].v) > vMax)
        index2 = np.argwhere(np.ravel(particles[i].v) < vMin)

        particles[i].v[0, index1] = vMax[index1]
        particles[i].v[0, index2] = vMin[index2]

        particles[i].x = particles[i].x + particles[i].v

        # Check positions

        index1 = np.argwhere(np.ravel(particles[i].x) > ub)
        index2 = np.argwhere(np.ravel(particles[i].x) < lb)

        particles[i].x[0, index1] = ub[index1]
        particles[i].x[0, index2] = lb[index2]

    print('Iteration #', j, '\n Swarm global best: ', s.gBestO,
          '\n ')
