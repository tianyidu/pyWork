import  threading
from time import ctime,sleep

loops = [2,4]


def loop(nloop,nsec):
	print('start loop ',nloop,' at ',ctime())
	sleep(nsec)
	print('end loop ',nloop,' at ',ctime())


class MysubThread(threading.Thread):
	def __init__(self,name,args,func):
		threading.Thread.__init__(self)
		self.args = args
		self.func = func
		self.name = name

	def run(self):
		print('thread ',self.name)
		self.func(*self.args)

class MyThread():
	def __init__(self,name,args,func):
		self.args = args
		self.func = func
		self.name = name

	def __call__(self):
		print('thread ',self.name)
		self.func(*self.args)

def main01():
	print('main start at ',ctime())
	nloops = range(len(loops))
	threads = []
	for i in nloops:
		#使用实例
		#t = threading.Thread(target=MyThread(loop.__name__, (i,loops[i]), loop))
		#使用继承
		t = MysubThread(loop.__name__, (i,loops[i]), loop)
		threads.append(t)

	for i in nloops:
		threads[i].start()

	for i in nloops:
		threads[i].join()

	print('all done at ',ctime())



def main():
	print('main start at ',ctime())
	threads = []
	nloops = range(len(loops))

	for i in nloops:
		t = threading.Thread(target=loop,args=(i,loops[i]))
		threads.append(t)

	for i in nloops:
		threads[i].start()

	for i in nloops:
		threads[i].join()

	print('all done at ',ctime())

if __name__ == '__main__':
	main01()