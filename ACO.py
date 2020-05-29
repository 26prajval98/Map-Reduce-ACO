import numpy as np
import random
import matplotlib.pyplot as plt
import sys


# ________________________________________________________________________________________________________________________________________________
# Class ACO start

class ACO():

	def __init__(self, vms, tasks, itr = 10, m = 10, inital_pheromone = 0.5, alpha = 1, beta = 1, rho = 0.5, Q = 100):
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
	# print("FCFS: machine id %d, makespan %d"  % (idx, mp))
	return mp


def rnd(vms, tasks):
	random = RANDOM(vms, tasks)
	mtt = random.run()
	idx, mp = make_span(mtt, vms, tasks)
	# print("RANDOM: machine id %d, makespan %d"  % (idx, mp))
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
	# print("ACO: machine id %d, makespan %d" % (idx, mp))
	return mp
	

def generate_random_array(size, a=30, b=100):
	return random.sample(range(a, b), size)


def multiobjective(m, n):
	vms = generate_randoms(m, 10, 20)
	tasks_multi_obj = []
	tasks_with_multi = []

	for i in range(n):
		tasks_multi_obj.append(generate_random_array(random.randint(1, 11)))
		tasks_with_multi.append([*tasks_multi_obj[-1]])
	
	indices = [-m  for i in range(n)]
	
	tasks = []
	
	i = 0
	while len(tasks_multi_obj):
		i %= len(tasks_multi_obj)
		next_idx = len(tasks)
		if next_idx - indices[i] < m:
			tasks.append(random.uniform(0, 0.1))
		else:
			tasks.append(tasks_multi_obj[i][0])
			indices[i] = len(tasks) - 1
			del tasks_multi_obj[i][0]
			if len(tasks_multi_obj[i]) == 0:
				del tasks_multi_obj[i]
				del indices[i]
				i-=1
			i += 1
	
	return vms, tasks, tasks_with_multi

def execute(m, n, a=10, b=20, c=30, d=100):
	vms = generate_randoms(m, a, b)
	tasks = generate_randoms(n, c, d)
	return execute_from_data(vms, tasks)

def execute_from_data(vms, tasks):
	return [fcfs(vms, tasks), rnd(vms, tasks), aco(vms, tasks)]


def is_none(l, length):
	if l is None:
		return [None for ii in range(length)]
	else:
		assert len(l), length
		return l


def plt_graphs(xss, yss, labels=None, colors=None, markers=None, x_label="x - axis", y_label="y - axis", title="A Graph"):
	l = len(xss)
	assert l, len(yss)

	labels = is_none(labels, l)
	markers = is_none(markers, l)
	colors = is_none(colors, l)

	for ll in range(l):
		plt.plot(xss[ll], yss[ll], label=labels[ll], color=colors[ll], marker=markers[ll])

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	plt.show()


def transpose(matrix):
    m = len(matrix)
    n = len(matrix[0])
    result = [[] for i in range(n)]
    for i in range(m+1):
        for j in range(n):
            if i == m:
                result[j].append(0)
            else:
                result[j].append(matrix[i][j])
    return result


def main():

	fcfs = []
	rnds = []
	aco = []
	tsk = []

	args = sys.argv

	if len(args) > 1:
		option = int(args[-1])
	else:
		option = 0
	
	if option == 0:
		vms = 10

		for tasks in range(10, 101, 10):
			ts = execute(vms, tasks)
			fcfs.append(ts[0])
			rnds.append(ts[1])
			aco.append(ts[2])
			tsk.append(tasks)

		makespans = [fcfs, rnds, aco]
		counts = [tsk, tsk, tsk]

		# print(fcfs)
		# print(rnds)
		# print(aco)
		# print(tsk)
		
		plt_graphs(counts, makespans, labels=["FCFS", "RANDOM", "ACO"], colors=["b", "g", "r"], markers=['.', 'o', '^'], x_label="Number of tasks", 
			y_label= "Makespans", title="Makespan with %d nodes (vms)"%(vms))

	elif option == 1:
		tasks = 100

		for vms in range(3, 15):
			ts = execute(vms, tasks)
			fcfs.append(ts[0])
			rnds.append(ts[1])
			aco.append(ts[2])
			tsk.append(vms)

		makespans = [fcfs, rnds, aco]
		counts = [tsk, tsk, tsk]
		
		print(fcfs)
		print(rnds)
		print(aco)
		print(tsk)

		plt_graphs(counts, makespans, labels=["FCFS", "RANDOM", "ACO"], colors=["b", "g", "r"], markers=['.', 'o', '^'], x_label="Number of nodes (vms)", 
			y_label= "Makespans", title="Makespan with %d tasks"%(tasks))

	elif option == 2:
		num_mach = 10
		for tsks in range(10, 101, 25):
			params = multiobjective(num_mach, tsks)
			vms = params[0]
			tasks = params[1]
			ts = execute_from_data(vms, tasks)
			fcfs.append(ts[0])
			rnds.append(ts[1])
			aco.append(ts[2])
			tsk.append(tsks)

		print("Done")
		makespans = [fcfs, rnds, aco]
		counts = [tsk, tsk, tsk]

		print(fcfs)
		print(rnds)
		print(aco)
		print(tsk)
		
		plt_graphs(counts, makespans, labels=["FCFS", "RANDOM", "ACO"], colors=["b", "g", "r"], markers=['.', 'o', '^'], x_label="Number of tasks", 
			y_label= "Makespans", title="Makespan with %d nodes (vms)"%(num_mach))

	elif option == 3:
		num_tasks = 100

		for vmns in range(5, 12):
			params = multiobjective(vmns, num_tasks)
			vms = params[0]
			tasks = params[1]
			ts = execute_from_data(vms, tasks)
			fcfs.append(ts[0])
			rnds.append(ts[1])
			aco.append(ts[2])
			tsk.append(vmns)
		
		print("Done")

		makespans = [fcfs, rnds, aco]
		counts = [tsk, tsk, tsk]

		print(fcfs)
		print(rnds)
		print(aco)
		print(tsk)
		
		plt_graphs(counts, makespans, labels=["FCFS", "RANDOM", "ACO"], colors=["b", "g", "r"], markers=['.', 'o', '^'], x_label="Number of tasks", 
			y_label= "Makespans", title="Makespan with %d nodes (tasks)"%(tasks))

	elif option == 4:
		
		num_mach=5
		num_tasks=30
		vms = generate_randoms(num_mach, 10, 20)
		tasks = generate_randoms(num_tasks, 30, 50)

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
		print(tasks)
		print(vms)
		print(mtt)
		print(idx, mp)

		mach_=list(range(num_mach))
		num_of_tasks=[]
		for _ in mach_:
			if _ in mtt.keys():
				num_of_tasks.append(len(mtt[_]))
			else:
				num_of_tasks.append(0)

		plt.bar(mach_,num_of_tasks)
		plt.xlabel('Machine ID')
		plt.ylabel('Number of tasks mapped')
		plt.title('Distribution of tasks')
		# plt.legend()
		plt.show()

	
	else:
		print("Provide argument as 0 or 1")

if __name__ == "__main__":
	main()
