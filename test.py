class test:

	def update(self, x):
		x['a'][0] = 200

	def run(self):
		x = {'a' : [100, 200, 300, 400]}
		self.update(x)
		print(x)

t = test()
t.run()