To run the file do the following:

python ACO.py [arg]

arg is the value which decides what experiment to run

if:

1. arg is 0 or not given : ACO, FCFS and Random algorithm will run and the result against them will be plotted for varying tasks given
	number of vms = 10
2. arg is 1 : ACO, FCFS and Random algorithm will run and the result against them will be plotted for varying vms given
	number of tasks = 100
3. arg is 2 : MO-ACO, MO-FCFS and MO-Random algorithm will run and the result against them will be plotted for varying tasks given
	number of vms = 10
4. arg is 3 : MO-ACO, MO-FCFS and MO-Random algorithm will run and the result against them will be plotted for varying vms given
	number of tasks = 100
5. arg is 4 : ACO vs MO-ACO for number of tasks = 100 and vms = 10
6. arg is 5 the lack of load balancing property is shown
