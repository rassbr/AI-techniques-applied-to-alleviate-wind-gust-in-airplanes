import numpy as np
def paramgen(param_choice,V_input):
	if param_choice == 1:
		modes = 2
		V = 100			 # airspeed [m/s]
		m = 100			# unit mass / area of wing
		y0 = np.zeros((modes*2,1))
		return V, m, y0
	if param_choice == 2:
		modes = 2
		Vmax = 180
		Vmin = 50
		V = (Vmax-Vmin)*np.random.rand() + Vmin
		m = 100			# unit mass / area of wing
		y0 = np.zeros((modes*2,1))
		return V, m, y0
		
	if param_choice == 3:
		modes = 2
		Vmax = 200
		Vmin = 50
		V = (Vmax-Vmin)*np.random.rand() + Vmin
		m = 100			# unit mass / area of wing
		y0 = np.zeros((modes*2,1))
		qmin = -1
		qmax = 1
		qdotmin = -1
		qdotmax = 1
		y0[0] = (qdotmax-qdotmin)*np.random.rand() + qdotmin
		y0[1] = (qdotmax-qdotmin)*np.random.rand() + qdotmin
		y0[2] = (qmax-qmin)*np.random.rand() + qmin
		y0[3] = (qmax-qmin)*np.random.rand() + qmin
		return V, m, y0
		
	if param_choice == 4:
		modes = 2
		Vmax = 200
		Vmin = 50
		V = (Vmax-Vmin)*np.random.rand() + Vmin
		mmin = 25
		mmax = 500
		m = 100			# unit mass / area of wing
		y0 = np.zeros((modes*2,1))
		return V, m, y0
		
	if param_choice == 5:
		modes = 2
		V = V_input
		m = 100			# unit mass / area of wing
		y0 = np.zeros((modes*2,1))
		return V, m, y0