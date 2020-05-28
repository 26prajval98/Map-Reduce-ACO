import numpy as np
import random
import matplotlib.pyplot as plt


# ________________________________________________________________________________________________________________________________________________
# Class ACO start

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

	def ET(self, t, m, offset):
		return self.tasks[t + offset]/self.vms[m]


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
			idx = idx[0][0]
			return idx
		

	# By tejas
	def updatePheromones(self, pheromone, times, soln):

		updatep={}
		for i in range(pheromone.shape[0]):
			v= {}
			for j in range(pheromone[i].shape[0]):
				v[j]=0.0

			updatep[i] = v
		
		for k in soln:
			updatevalue= self.Q/times[k]
			tour = soln[k]
			del tour[-1]

			for i in range(pheromone.shape[0]):
				v = {}
				for j in range(pheromone[i].shape[0]):
					if j in tour.values():
						v[j] = updatep[i][j] + updatevalue
					else:
						v[j] = updatep[i][j]
				updatep[i] = v

		for i in range(pheromone.shape[0]):
			x = pheromone[i]

			for j in range(pheromone[i].shape[0]):
				pass
				x[j]=(1-self.rho)*x[j]+updatep[i][j]
			pheromone[i] = x


	def globalUpdatePheromone(self, pheromone, mn, soln, offset):
		updatevalue = self.Q/mn

		for i in soln:
			#  taskid : mid
			v = pheromone[i - offset]
			pheromone[i - offset, soln[i]] += updatevalue


	def ACO(self, tasks, offset):
		opt_sol = {}
		mn_idx = 0

		et = np.zeros((len(tasks.keys()),len(self.vms.keys())))

		n = len(tasks.keys())

		for i in range(len((tasks.keys()))):
			for j in range(len((self.vms.keys()))):
					et[i,j] = self.ET(i,j, offset)
		
		pheromone = np.full((len(tasks.keys()),len(self.vms.keys())), self.inital_pheromone)

		for i in range(self.itr):
			# shuffle machines to assign it to ants. no of ants = no. of machines
			soln = {}
			machines = list(range(len(self.vms.keys())))
			random.shuffle(machines)
			times = []

			for k in range(self.m):
				soln[k] = {}
				# assign no task to machine k intially
				soln[k][-1] =  machines[k]
				mx = 0
				for task in tasks: 
					vm = self.chooseVM(et[task - offset], pheromone[task - offset], soln[k])
					soln[k][task] = vm
					
					# changed to max
					mx = max(mx, et[task - offset, vm]) 
				
				times.append(mx)
			
			mn_idx, mn = times.index(min(times)), min(times)

			self.updatePheromones(pheromone, times, soln)
			self.globalUpdatePheromone(pheromone, mn, soln[mn_idx], offset)
		
		opt_sol = soln[mn_idx]

		return opt_sol

	
	def run(self):
		'''
			vms [100MIPS]
			tasks [st, inst count]
		'''

		m = len(self.vms.keys())
		n = len(self.tasks.keys())

		allocatedtasks={}

		# Changed n/m - 1
		for i in range((int)(n/(m-1))):
			subtasks={}
			for j in list(self.tasks.keys())[i*(m-1):(i+1)*(m-1)]:
				subtasks[j]=self.tasks[j]
			at = self.ACO(subtasks, i * (m-1))

			for j in at:
				allocatedtasks[j]=at[j]

		subtasks={}
		for j in range((len(self.tasks.keys())//(m-1))*(m-1),len(self.tasks.keys())):
			subtasks[j]=self.tasks[j]

		if len(subtasks.keys()):
			at = self.ACO(subtasks, (len(self.tasks.keys())//(m-1))*(m-1))

			for j in at:
				allocatedtasks[j]=at[j]

		return allocatedtasks
# ________________________________________________________________________________________________________________________________________________
# Class ACO end

# ________________________________________________________________________________________________________________________________________________
# Class FCFS start

class FCFS():

	def __init__(self, vms, tasks):
		self.vms = {}
		self.tasks = {}

		for i in range(len(vms)):
			self.vms[i] = vms[i]

		for i in range(len(tasks)):
			self.tasks[i] = tasks[i]
	

	def run(self):
		soln = {}
		m = len(self.vms.keys())
		mid = 0
		
		for task in self.tasks:
			if mid in soln:
				soln[mid].append(task)
			else:
				soln[mid] = [task]

			mid += 1
			mid %= m

		return soln

# ________________________________________________________________________________________________________________________________________________
# Class FCFS end


# ________________________________________________________________________________________________________________________________________________
# Class Random start
class RANDOM():

	def __init__(self, vms, tasks):
		self.vms = {}
		self.tasks = {}

		for i in range(len(vms)):
			self.vms[i] = vms[i]

		for i in range(len(tasks)):
			self.tasks[i] = tasks[i]
	

	def run(self):
		soln = {}
		m = len(self.vms.keys())
		
		for task in self.tasks:
			mid = random.choice(range(m))
			if mid in soln:
				soln[mid].append(task)
			else:
				soln[mid] = [task]

		return soln

# ________________________________________________________________________________________________________________________________________________
# Class Random end


def generate_randoms(n, i, f):
	return np.random.choice(list(range(i, f)), n)


def calculate_time(mtt, vms, tasks):
	times = {}
	
	for m in mtt:
		times[m] = 0
		for i  in range(len(mtt[m])):
			times[m] += (tasks[mtt[m][i]]/vms[m])
	return times


def make_span(mtt, vms, tasks):
	times = calculate_time(mtt, vms, tasks)
	max_time_machine = max(times, key=times.get)
	max_time = times[max_time_machine]
	return max_time_machine, max_time


def fcfs(vms, tasks):
	fcfc = FCFS(vms, tasks)
	mtt = fcfc.run()
	idx, mp = make_span(mtt, vms, tasks)
	print("FCFS: machine id %d, makespan %d"  % (idx, mp))
	return mp


def rnd(vms, tasks):
	random = RANDOM(vms, tasks)
	mtt = random.run()
	idx, mp = make_span(mtt, vms, tasks)
	print("RANDOM: machine id %d, makespan %d"  % (idx, mp))
	return mp
	

def aco(vms, tasks):
	aco = ACO(vms, tasks)
	ttm = aco.run()
	mtt = {}
	
	for t in ttm:
		if ttm[t] in mtt:
			mtt[ttm[t]].append(t)
		else:
			mtt[ttm[t]] = []
			mtt[ttm[t]].append(t)
	
	idx, mp = make_span(mtt, vms, tasks)
	print("ACO: machine id %d, makespan %d" % (idx, mp))
	return mp
	

def execute(m, n, a=10, b=20, c=30, d=100):
	vms = generate_randoms(m, a, b)
	tasks = generate_randoms(n, c, d)

	return [fcfs(vms, tasks), rnd(vms, tasks), aco(vms, tasks)]


def is_none(l, length):
	if l is None:
		return [None for ii in range(length)]
	else:
		assert len(l), length
		return l


def plt_graphs(xss, yss, labels=None, colors=None, markers=None, x_label="x - axis", y_label="y - axis"):
	l = len(xss)
	assert l, len(yss)

	labels = is_none(labels, l)
	markers = is_none(markers, l)
	colors = is_none(colors, l)

	for ll in range(l):
		plt.plot(xss[ll], yss[ll], label=labels[ll], color=colors[ll], marker=markers[ll])
	plt.legend()
	plt.show()


def main():
	plot = []

	for tasks in range(10, 101, 10):
		plot.append(execute(10, tasks))


if __name__ == "__main__":
	main()
