import numpy as np
import random

class ACO():

	def __init__(self, vms, tasks, itr = 2, m = 10, inital_pheromone = 0.5, alpha = 1, beta = 5, rho = 0.5, Q = 100):
		self.vms = {}
		self.tasks = {}

		for i in range(len(vms)):
			self.vms[i] = vms[i]

		for i in range(len(tasks)):
			self.tasks[i] = tasks[i]
		
		self.itr = itr
		self.inital_pheromone = inital_pheromone
		self.alpha = alpha
		self.beta = beta
		self.m = m < len(vms) and m or len(vms)
		self.Q = Q
		self.rho = rho

	def ET(self, t, m):
		return self.tasks[t]/self.vms[m]


	def FCFS_OPT(self):
		pass

	
	def chooseVM(self, et, pheromone, soln):
		# check probability of a each machine to perform the task
		prob = np.zeros(len(et))
		total = 0

		for machine in range(len(et)):
			if not machine in soln.values():
				exec = et[machine]
				pher = pheromone[machine]
				prob[machine] = ((1/exec)**self.beta)*(pher**self.alpha)
				total += prob[machine]

		if total:
			prob = prob / total
		
		max_prob = prob.max()

		if max_prob <= 0:
			return -1
		else:
			idx = np.where(np.isclose(prob, max_prob))
			idx = idx[0]
			return idx
		

	# By tejas
	def updatePheromones(self,pheromone, times, soln):

		updatep={}
		for i in range(pheromone.shape[0]):
			v= {}
			for j in range(pheromone[i].shape[0]):
				v[j]=0.0

			updatep[i]=v
		
		for k in range(len(soln)):
			# what is Q
			updatevalue= self.Q/times[k]
			tour=soln[k]
			del tour[-1]

			for i in range(pheromone.shape[0]):
				v={}
				for j in range(pheromone[i].shape[0]):
					if j in tour.values():
						v[j]=updatep[i][j]+updatevalue

					else:
						v[j]=updatep[i][j]
				updatep[i]=v
			
		
		for i in range(pheromone.shape[0]):
			x=pheromone[i]

			for j in range(pheromone[i].shape[0]):
				pass
				# print(x[j], updatep[i][j])
				x[j]=(1-self.rho)*x[j]+updatep[i][j]
			pheromone[i]=x



	def globalUpdatePheromone(self,pheromone,mn,soln):
		updatevalue=self.Q/mn

		for i in range(len(soln)-1):
			v= pheromone[i]
			v[soln[i]]=v[soln[i]]+updatevalue
			pheromone[i]=v


	def ACO(self):
		
		opt_sol = {}
		mn_idx = 0

		et = np.zeros((len(self.tasks.keys()),len(self.vms.keys())))

		n = len(self.tasks.keys())

		for i in range(len((self.tasks.keys()))):
			for j in range(len((self.vms.keys()))):
					et[i,j] = self.ET(i,j)
		
		pheromone = np.full((len(self.tasks.keys()),len(self.vms.keys())), self.inital_pheromone)

		for i in range(self.itr):
			# shuffle machines to assign it to ants. no of ants = no. of machines
			soln = {}
			machines = np.random.choice(range(len(self.tasks.keys())), self.m, replace=False)
			times = []

			for k in range(self.m):
				soln[k] = {}
				# assign no task to machine k intially
				soln[k][-1] =  machines[k]
				mx = 0

				for task in self.tasks:
					vm = self.chooseVM(et[task], pheromone[task], soln[k])
					soln[k][task] = vm
					mx = mx > et[task, vm] and mx or et[task, vm]
				
				times.append(mx)
			
			mn_idx, mn = times.index(min(times)), min(times)

			self.updatePheromones(pheromone, times, soln)
			self.globalUpdatePheromone(pheromone, mn, soln[mn_idx])
		
		opt_sol = soln[mn_idx]

		return opt_sol

	
	def run(self):
		'''
			vms [100MIPS]
			tasks [st, inst count]
		'''

		m = len(self.vms.keys())
		n = len(self.tasks.keys())

		# while len(self.tasks.keys()):
		if m >= n:
			return self.FCFS_OPT()
		else:
			return self.ACO()

def generate_randoms(n, i, f):
	return random.sample(range(i, f), n)


def main():
	vms = generate_randoms(3, 10, 20)
	tasks = generate_randoms(10, 30, 50)
	ac = ACO(vms, tasks)
	print(ac.run())

if __name__ == "__main__":
	main()