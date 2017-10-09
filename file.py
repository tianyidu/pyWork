import os
import os.path
import shutil
import time,datetime

#拷贝指定文件到目标目录
def moveFileto(sourceDir, targetDir): 
	shutil.copy(sourceDir, targetDir)

#列出某一目录下文件
def listDirPath(dirPath):
	print(os.listdir(dirPath))

#拷贝指定目录下所有文件到目标路径
def copyFiles(sourceDir, targetDir): 
	#print("outer sourceDir is "+sourceDir)
	if sourceDir.find(".svn") > 0: 
		#print("inner sourceDir is "+sourceDir)
		return
	for file in os.listdir(sourceDir):
		#print("begin copy file getfileName:"+file)
		sourceFile = os.path.join(sourceDir, file)
		targetFile = os.path.join(targetDir, file) 
		#print("begin copy file :"+file)
		if os.path.isfile(sourceFile): 
			if not os.path.exists(targetDir): 
				os.makedirs(targetDir) 
			if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))): 
				open(targetFile, "wb").write(open(sourceFile, "rb").read()) 
		if os.path.isdir(sourceFile): 
			First_Directory = False
			copyFiles(sourceFile, targetFile)

#删除指定目录下所有文件 flag为true时，递归删除
def deleteDirFiles(sourceDir,flag=False):
	for file in os.listdir(sourceDir):
		sourceFile=os.path.join(sourceDir,file)
		if os.path.isfile(sourceFile):
			print("delete file :"+sourceFile)
			os.remove(sourceFile)
		if os.path.isdir(sourceFile) and flag == True:
			print("*******delete dir :"+sourceFile)
			deleteDirFiles(sourceFile,flag)
			os.rmdir(sourceFile)
			print("*******end dir :"+sourceFile)

if __name__ == "__main__":
  #moveFileto("D:/wing/wing/temp/file.log", "D:/wing/wing/temp/file.test.log")
  #listDirPath("D:/wing/wing/temp")
  #copyFiles("D:/wing/wing/temp/SyncNode","D:/wing/wing/temp1")
  deleteDirFiles("D:/wing/wing/temp1",True)