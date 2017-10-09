import os
import sys

from linux import *


if __name__ == "__main__":

	linux = Linux()
	user = os.environ["username"]
	while True:
		currentdir = linux.getcurdir()		
		operators = input("["+user+"@"+currentdir+"] ")
		operator = operators.split()
		if operator[0]=="exit":
			break

		if hasattr(linux,operator[0]):
			sysoper = getattr(linux,operator[0])
			operator.pop(0)
		#	print(len(operator),operator)
			sysoper(operator)
