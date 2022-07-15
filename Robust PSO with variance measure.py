import numpy as np
import matplotlib.pyplot as plt


# def objectiveFunction(x):
#     return np.sum(np.square(x))

def objectiveFunction(x):

    # vars
    i = 0.001
    H = 50  # the number of sample points
    delta = 0.05  # error
    threshold = 0.1  # threshold for the variance measure, 1 is less robust and closer to zero gets more robust

    # Calculate the objective
    f = -(1/(np.sqrt(2*np.pi))*np.exp(-(np.square(x[0, 0]-1.5) + np.square(x[0, 1]-1.5))) +
          (2/np.sqrt(2*np.pi))*np.exp(-0.5*((np.square(x[0, 0]-0.5)+np.square(x[0, 1]-0.5))/i)))

    F = []

    for k in range(0, H):
        mError = 2*delta*np.random.random()-delta

        x_perturbed = x + mError
        F.append(-(1/(np.sqrt(2*np.pi))*np.exp(-(np.square(x_perturbed[0, 0]-1.5) + np.square(x_perturbed[0, 1]-1.5))) +
                 (2/np.sqrt(2*np.pi))*np.exp(-0.5*((np.square(x_perturbed[0, 0]-0.5)+np.square(x_perturbed[0, 1]-0.5))/i))))

    variance = np.abs(np.sum(F) - f) / np.abs(f)

    if variance < threshold:
        return f
    else:
        o = 2000

    return (np.sum(F) + o)/(H+1)

# Define the details of the objective function

# vars = 10
# ub = 10*np.ones(vars)
# lb = -10*np.ones(vars)


vars = 2
ub = 7 * np.ones(vars)
lb = 2 * np.ones(vars)

# Define PSO's parameters

numberParticles = 30
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
          '\nGlobal Best: ', s.gBestX, '\n')
