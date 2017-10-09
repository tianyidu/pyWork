import os

class Linux:

	def pwd(self,operator=[]):
		if len(operator)>1:
			self.__usage("pwd",None)
		else:
			print(os.getcwd())

	def cd(self,operator):
		#print(__name__)
		if len(operator)>1:
			self.__usage("cd","path")
		else:
			os.chdir(operator[0])
			print(os.listdir())

	def ls(self,operator):
		if len(operator)>1:
			self.__usage("ls","path")
		else:
			if len(operator)==0:
				print(os.listdir())
			elif os.path.isdir(operator[0]):
				print(os.listdir(operator[0]))
			else:
				self.__usage("ls","dir")

	def cp(self,operator):
		if len(operator)<2:
			self.__usage("cp", "sourcefile target file")
		else:
			sourcefile = operator[0]
			sourcedirsuff = sourcefile.split(os.sep)
			targetfile = os.path.join(operator[1],sourcedirsuff[-1])
			if os.path.isdir(sourcefile):
				self.mkdir(targetfile)
				files = os.listdir(sourcefile)
				for file in files:
					self.cp([os.path.join(sourcefile,file),targetfile])
			elif os.path.isfile(sourcefile):
				if not os.path.exists(targetfile):
					with open(sourcefile,"rb") as sf:
						open(targetfile,"wb").write(sf.read())

	def mkdir(self,operator):
		if isinstance(operator,list):
			for dir in operator:
				if os.path.exists(dir):
					print("directory",dir," already exists")
				else:
					print("directory",dir,"created")
					os.mkdir(dir)
		else:
			print("directory",operator,"created")
			os.mkdir(operator)

	def rm(self,operator):
		for file in operator:
			if os.path.isfile(file) and os.path.exists(file):
				os.remove(file)
			else:
				self.__usage("rm", "file")

	def __rmdirfile(self,dir):
		files = os.listdir(dir)
		subfiles = []
		for file in files:
			file = os.path.join(dir,file)
			if os.path.isdir(file):
				self.__rmdirfile(file)
			elif os.path.isfile(file):
				subfiles.append(file)
			else:
				print("rmdirfile",file,"failed")

		self.rm(subfiles)
		os.rmdir(dir)

	def rmdir(self,operator):
		for dir in operator:
			if os.path.isdir(dir):
				self.__rmdirfile(dir)
			else:
				self.__usage("rmdir","dir")

	def __usage(self,funcname,argument):
		print("Usage:"+funcname+" "+argument)

	def getcurdir(self):
		currentdirs = os.getcwd().split(os.path.sep)
		#print("current dir", currentdirs)
		currentdir = currentdirs[-1]
		if currentdir=="":
			currentdir = currentdirs[-2]

		return currentdir