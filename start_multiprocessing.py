from multiprocessing import Process, Queue
import sys

# from generative_model_old import *
from generative_model import *
from game import *

threads = 0

def run_multiprocessing():
	action = Queue()

	p1 = Process(target = run_generative_model, args = (action,))
	p1.start()
	p2 = Process(target = run_game, args = (action,))
	p2.start()

def run_generative_model(action):
	global threads
	while threads < sys.maxsize:
		threads += 1 
		start_generative_model(action)
		

def run_game(action):
	global threads
	while threads < sys.maxsize:
		threads += 1 
		start_game(action)

if __name__ == '__main__':
	run_multiprocessing()