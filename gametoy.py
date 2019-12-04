import numpy as np
import matplotlib.pyplot as plt
from reward import reward_calc
from parameters import paramgen

def main_Parameters(param_choice,V_input):
	# System Parameters
	#airplane condition
	V,m,y0 = paramgen(param_choice, V_input)
	#wing parameters
	s = 7.5			# semi span
	b = 1.0			# semi chord
	c = 2*b			# chord
	xf = 0.48 * c	  # position of elastic axis from leading edge
	xcm = 0.5 * c	  # position of centre of mass from leading edge

	e = xf / c - 0.25  # eccentricity between elastic axis and aero center (1/4 chord)
	# 'e' is the distance between the elastic center and aerodynamic center,
	# positive when elastic axis is behind the latter)
	aw = 2 * np.pi		# lift curve slope ( how rapidly the wing generates lift with change in AOA.)
	#bw = e * aw		# control surface lift curve slope
	EE = 0.1	 # fraction of chord made up by control surface

	rho = 1.225		# air density
	Mtdot = -1.2	   # unsteady torsional damping term
	#damping_Y_N = 0	# =1 if damping included =0 if not included

	bending_freq = 5   # bending freq in Hz - approximate - ignores coupling term - K
	torsion_freq = 10  # torsion freq in Hz - approximate - ignores coupling term - Theta

	modes = 2
	#time parameters
	dt = 0.001				 #simulation timestep
	tmin = 0 
	tmax = 10			# simulatino start / end
	t = np.arange(tmin,tmax,dt)			  # Column vector
	npts = np.size(t)

	#gust parameters
	gust_amp_1_minus_cos = 10	 # Max velocity of (1 - cosine) gust   (m/s)
	gust_t = .05				 # Fraction of total time that is gust (0 - 1)

		# Matrices

	# Inertia matrix A
	A = np.zeros((modes,modes))
	A[0,0] = (m * s * c) / 5
	A[0,1] = m * s / 4 * (c**2 / 2 - c * xf)
	A[1,0] = A[0,1]
	A[1,1] = m * s / 3 * (c**3 / 3 - c**2 * xf + xf**2 * c)

	# Stiffness matrix E
	E = np.zeros((modes,modes))
	EI = (bending_freq * np.pi * 2)**2 * A[0,0] / 4 * s**3   # q1 term - bending stiffness
	GJ = (torsion_freq * np.pi * 2)**2 * A[1,1] * s		 # q2 term - torsion stiffness
	E[0,0] = 4 * EI / s**3
	E[1,1] = GJ / s

	# Aerodynamic damping matrix
	B = np.zeros((modes,modes))
	B[0,0] = c * s * aw / 10
	B[0,1] = 0
	B[1,0] = -c**2 * s * e*aw / 8
	B[1,1] = -c**3 * s * Mtdot / 24


	# Aerodynamic stiffness matrix
	C = np.zeros((modes,modes))
	C[0,0] = 0
	C[0,1] = c * s * aw / 8
	C[1,0] = 0
	C[1,1] = -c**2 * s * e*aw / 6


	# Structural Damping
	D=0*C

		# Control Surface
	#EE = 0
	EE = 0.1	 # fraction of chord made up by control surface
	ac = aw / np.pi * (np.arccos(1 - 2 * EE) + 2 * np.sqrt(EE * (1 - EE)))
	bc = -aw / np.pi * (1 - EE) * np.sqrt(EE * (1 - EE))
	F_control = rho * V**2 * np.array([-c * s * ac / 6, c**2 * s * bc / 4])
	# gust
	# Gust vector
	#F_gust = rho*V*c*s*[s/4 c/2]'
	F_gust = rho * V * np.array([-aw * s * c / 6,aw*e * c**2 * s / 4])
	Sgust = np.zeros(np.size(t))
	g_end = tmax * gust_t
	gt = np.sum(t < g_end)
	ft = t

	if gust_amp_1_minus_cos != 0:
		for ii in range(0,gt):
			Sgust[ii] = gust_amp_1_minus_cos / 2 * (1 - np.cos(2 * np.pi * t[ii] / g_end))

	invA = np.linalg.inv(A)
	Q = np.zeros((2*modes,2*modes))
	Q[0,:] = [0,0,1,0]
	Q[1,:] = [0,0,0,1]
	Q[2:4,0:2] = np.dot(-invA,(rho*V**2*C+E))
	Q[2:4,2:4] = np.dot(-invA,rho*V*B+D)

	Beta = np.zeros((modes*2,1))
	Beta[2:4,0] = np.dot(invA,F_control)
	Gust = np.zeros((modes*2,1))
	Gust[2:4,0] = np.dot(invA,F_gust)
	beta = 0
	

	n = int(np.ceil(tmax/dt))
	
	return Q, Beta, beta, Gust, tmax, dt, n, t, Sgust, y0, s, xf, V

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self,reward_choice = 1, param_choice = 1, V_input = 100):
		self.Q, self.Beta, self.beta, self.Gust, self.tmax, self.dt, self.n, self.t, self.Sgust, self.y, self.s, self.xf, self.V = main_Parameters(param_choice,V_input)
		self.x = 0
		self.observation_space = 4
		self.action_space = 3
		self.vy = np.zeros((self.n,self.observation_space))
		self.actions = np.zeros((self.n))
		self.betas = np.zeros((self.n))
		self.i = 0
		self.counter = 0
		self.terminal = 0
		self.gameover = 0
		self.score = 0
		self.reward = 0
		self.delta_ang = 0.01
		self.betalim = 20
		self.reward_choice = reward_choice
		self.param_choice = param_choice
		self.V_input = V_input
		self.z = 0
		self.zref = np.zeros((self.n))
		self.Z = np.zeros((self.n))

	def step(self, action):
		if self.i+1 < self.n:
			if action == 1:
				action = 1
			elif action == 2:
				action = -1
			else:
				action = 0
			action = action*self.delta_ang
			self.counter = self.counter+1
			self.beta = self.beta+action
			if self.beta > 20:
				self.beta = 20
				#return 0
			elif self.beta < -20:
				self.beta = -20
				#return 0
			self.betas[self.i] = self.beta
			self.x,self.y = self.rk4step(self.x,self.y,self.dt)
			self.actions[self.i] = action
			self.vy[self.i,0] = self.y[0]
			self.vy[self.i,1] = self.y[1]
			self.vy[self.i,2] = self.y[2]
			self.vy[self.i,3] = self.y[3]
			self.z = self.y[0]*self.s**2 - self.y[1]*self.s*self.xf
			self.Z[self.i] = self.z
			reward, self.gameover = reward_calc(self.vy[self.i,:],self.vy[self.i-1,:],self.z, self.zref[self.i],self.beta,self.betalim,self.reward_choice)
			self.reward = reward
			self.i = self.i+1
			if np.abs(self.y[0]) > .15 or np.abs(self.y[1]) > .15 or np.abs(self.y[2]) > 4 or np.abs(self.y[3]) > 4:
				self.terminal = 1
				self.reward = -10
			self.score = self.score + self.reward
			return [self.y, self.reward, self.terminal, self.score]
		else:
			self.terminal = 1
			self.reward = 10
			self.score = self.score + self.reward
			#print('Done')
			return [self.y, self.reward, self.terminal, self.score]

	def reset(self):
		counter = self.counter
		reward_choice = self.reward_choice
		param_choice = self.param_choice
		V_input = self.V_input
		zref = self.zref
		self.__init__(reward_choice,param_choice, V_input)
		self.counter = counter
		self.zref = zref
		return self.y
		#self.reward_choice = reward_choice
		#self.param_choice = param_choice

	def render(self, mode='human', close=False):
		plt.title('q1 and q2 in time')
		plt.plot(self.t,self.vy[:,0],self.t,self.vy[:,1])
		plt.gca().legend(('q1','q2')) 
		plt.show()
		plt.title('qdot1 and qdot2 in time')
		plt.plot(self.t,self.vy[:,2],self.t,self.vy[:,3])
		plt.gca().legend(('qdot1','qdot2'))
		plt.show()
		plt.title('Wing tip Leading Edge Displacement (m)')
		plt.plot(self.t,self.Z,self.t,self.zref)
		plt.gca().legend(('Control','NoControl'))
		plt.show()
		plt.plot(self.t,self.betas)
		plt.title('Surface control Position')
		plt.show()

	def plot(self, filename, mode='human', close=False):
		plt.plot(self.t,self.vy[:,0],self.t,self.vy[:,1])
		plt.gca().legend(('q1','q2'))
		plt.title('q1 and q2 in time')
		plt.savefig(filename+'.png')
		plt.close()
		plt.plot(self.t,self.vy[:,2],self.t,self.vy[:,3])
		plt.gca().legend(('qponto1','qponto2'))
		plt.title('qponto1 and qponto2 in time')
		plt.savefig(filename+'_ponto.png')
		plt.close()
		plt.plot(self.t,self.betas)
		plt.title('Surface control Position')
		plt.savefig(filename+'_rewards.png')
		plt.close()
		plt.plot(self.t,self.Z,self.t,self.zref)
		plt.gca().legend(('Control','NoControl'))
		plt.title('Wing tip Leading Edge Displacement (m)')
		plt.savefig(filename+'_Z.png')
		plt.close()

	def f(self,x,y):
		nowgust = np.interp(x,self.t,self.Sgust);
		#print(self.Q.dot(y) + self.Beta*self.beta + self.Gust*nowgust);
		return self.Q.dot(y) + self.Beta*self.beta + self.Gust*nowgust;

	def rk4step(self,x0, y0, h):
		x = x0
		y = y0
		k1 = h * self.f(x, y)
		k2 = h * self.f(x + 0.5 * h, y + 0.5 * k1)
		k3 = h * self.f(x + 0.5 * h, y + 0.5 * k2)
		k4 = h * self.f(x + h, y + k3)
		vx = x = x0 +  h
		vy = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
		return vx, vy

	def zrefer(self):
		i = 0
		while self.terminal == 0:
			self.step(0)
			self.zref[i] = self.z
			i += 1
		self.resetzrefer()

	def resetzrefer(self):
		self.x = 0
		self.vy = np.zeros((self.n,4))
		self.actions = np.zeros((self.n))
		self.betas = np.zeros((self.n))
		self.i = 0
		self.terminal = 0
		self.gameover = 0
		self.score = 0
		self.reward = 0
		self.z = 0
		self.beta = 0
		self.y = np.zeros((2*2,1))

	def stepbeta(self, beta):
		if self.i+1 <self.n:
			self.counter = self.counter+1
			self.beta = beta
			if self.beta > 20:
				self.beta = 20
				#return 0
			elif self.beta < -20:
				self.beta = -20
				#return 0
			self.betas[self.i] = self.beta
			self.x,self.y = self.rk4step(self.x,self.y,self.dt)
			self.i = self.i+1
			self.actions[self.i] = self.beta-beta
			self.vy[self.i,0] = self.y[0]
			self.vy[self.i,1] = self.y[1]
			self.vy[self.i,2] = self.y[2]
			self.vy[self.i,3] = self.y[3]
			self.z = self.y[0]*self.s**2 - self.y[1]*self.s*self.xf
			self.Z[self.i] = self.z
			reward, self.gameover = reward_calc(self.vy[self.i,:],self.vy[self.i-1,:],self.z, self.zref[self.i],self.beta,self.betalim,self.reward_choice)
			self.score = self.score + reward
			self.reward = reward
			return [self.y, self.reward, self.terminal, self.score]
		else:
			self.terminal = 1
			#print('Done')
			return [self.y, self.reward, self.terminal, self.score]