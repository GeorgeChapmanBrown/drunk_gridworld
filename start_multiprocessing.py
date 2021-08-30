from multiprocessing import Process, Queue
import sys

from generative_model_old import *
# from generative_model import *
from game import *

threads = 0

def run_multiprocessing():
	action = Queue()
	drunk = Queue()

	p1 = Process(target = run_generative_model, args = (action, drunk,))
	p1.start()
	p2 = Process(target = run_game, args = (action, drunk,))
	p2.start()

def run_generative_model(action, drunk):
	global threads
	while threads < sys.maxsize:
		threads += 1 
		start_generative_model(action, drunk)
		

def run_game(action, drunk):
	global threads
	while threads < sys.maxsize:
		threads += 1 
		start_game(action, drunk)

if __name__ == '__main__':
	run_multiprocessing()