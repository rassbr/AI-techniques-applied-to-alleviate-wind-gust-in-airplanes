import numpy as np
def reward_calc(y,y0,z,zref,beta,betalim,reward_choice,step_i = 0):
	if reward_choice == 1:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		reward = 0
		if beta>betalim or beta<-betalim:
			reward = -250
			gameover = 1
			return reward, gameover
		if y[0] > 2 or y[0]<-2:
			reward += -1
		if y[1] > 2 or y[1]<-1:
			reward += -1
		r = -(np.abs(y[:2]) - np.abs(y0[:2]))
		reward += np.sum(np.sign(r))/2
		return reward, gameover
	if reward_choice == 2:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		reward = 0
		if beta>betalim or beta<-betalim:
			reward = -250
			gameover = 1
			return reward, gameover
		r = np.abs(y[:2]) - np.abs(y0[:2])
		temp = np.floor((1/np.sum(r))/1000)
		temp = np.sign(temp)*min(np.abs(temp),10)
		reward += temp
		return reward, gameover
		
	if reward_choice == 3:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		reward = 0
		if beta>betalim or beta<-betalim:
			reward = -250
			gameover = 1
			return reward, gameover
		reward += 5-np.abs(z)
		return reward, gameover

	if reward_choice == 4:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		reward = 0
		if beta>betalim or beta<-betalim:
			reward = -250
			gameover = 1
			return reward, gameover
		reward += np.abs(zref)-np.abs(z)
		return reward, gameover
		
	if reward_choice == 5:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		reward = 0
		
		r = -(np.abs(y[:]) - np.abs(y0[:]))
		reward = r[1]*7.5**2 + r[2]*7.5*.98
		return reward, gameover

	if reward_choice == 6:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		r = 1/np.abs(z)
		if r > 5:
			reward = 5
		elif r < .1:
			reward = .1
		else:
			reward = r
		return reward, gameover

	if reward_choice == 7:
		# y-> actual state
		# y0-> previous state
		# beta-> angle of control surface
		gameover = 0
		r = 1/np.abs(z)
		if r > 5:
			reward = 5
		elif r < .1:
			reward = .1
		else:
			reward = r
		if step_i < 1000:
			reward = reward + 1
		if step_i > 1000:
			reward = reward + 2.5
		return reward, gameover