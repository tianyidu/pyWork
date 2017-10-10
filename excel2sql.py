#encoding:utf-8

import xlrd
import os
"""读取excel"""
"""""""
# def readFromXl(xlName):
# 	xl = xlrd.open_workbook(xlName)
# 	xlSheet = xl.sheets()
# 	for sheet in xlSheet:
# 		print((sheet.name))

# class xlsxObj():

# 	def __init__(self,xlsName):
# 		self.xlsName = xlsName

# 	def readFromxls(self):
# 		try:
# 			self.xlsBook = xlrd.open_workbook(self.xlsName)
# 		except Exception as e:
# 			print(e) 
# 			self.xlsBook = None

# 	def getSheetByName(self,sheetName):
# 		if self.xlsBook:
# 			self.xlsBook.sheet_by_name(sheetName)


class GetSqlObj():
	def __init__(self,xlsSheet,filePath):
		self.xlsSheet = xlsSheet
		self.sql = ""
		self.comment = ""
		self.filePath = filePath
		
	def getSql(self,startrow=2):

		for i in range(startrow,self.xlsSheet.nrows):
			#print(i,self.xlsSheet.nrows)
			if self.xlsSheet.cell(i,0):
				if i==2:
					tableName = self.xlsSheet.cell_value(i,1)
					self.sql = "create table "+tableName+"("
					self.comment = ""
				elif self.xlsSheet.cell_value(i-1,0)=="" and self.xlsSheet.cell_value(i,0):
					tableName = self.xlsSheet.cell_value(i,1)
					self.sql = self.sql+"create table "+tableName+"("
					self.comment = ""

			colName = self.xlsSheet.cell_value(i,2)
			self.sql = self.sql+"\t"+colName+" "+self.xlsSheet.cell_value(i,4)
			self.comment = self.comment+"comment on column "+tableName+"."+colName+" is '"+self.xlsSheet.cell_value(i,3)+"';\n"

			if i==self.xlsSheet.nrows-1:
				self.sql = self.sql+");\n"
				self.sql = self.sql+self.comment+"\n"
			elif self.xlsSheet.cell_value(i+1,4)=="":
				self.sql = self.sql+");\n"
				self.sql = self.sql+self.comment+"\n"
				break
			elif self.xlsSheet.cell_value(i+1,0)!="" and self.xlsSheet.cell_value(i,0)=="":
				self.sql = self.sql+");\n"
				self.sql = self.sql+self.comment+"\n"				
			else:
				self.sql = self.sql+",\n"
	
	def getDropSql(self,startrow=2):
		for i in range(startrow,self.xlsSheet.nrows):
			if self.xlsSheet.cell_value(i,4)=="":
				print(self.xlsSheet.name,"break")
				break
			if i==2:
				self.sql = "drop table "+self.xlsSheet.cell_value(i,1)+";\n"
			elif self.xlsSheet.cell_value(i,0) and self.xlsSheet.cell_value(i-1,0)=="":
				self.sql = self.sql+"drop table "+self.xlsSheet.cell_value(i,1)+";\n"

	def writeSql(self,drop=None):
		if drop:
			file = "drop_"+self.xlsSheet.name+(".sql")
		else:
			file = self.xlsSheet.name+(".sql")

		self.fileName = os.path.join(self.filePath,file)
		print(self.fileName)

		with open(self.fileName,"w") as f:
			f.write(self.sql)

	def __str__(self):
		return self.sql	





if __name__=="__main__":
	#readFromXl("")
	# xlsx = xlsxObj("")
	# xlsx.readFromxls()

	# tableNameCol = 0 表名所在列
	# colNameCol = 2    表字段名所在列
	# typeNameCol = 4  表字段类型所在列
	# commentCol = 3  表字段注释所在列

	sheets = [""]

	xlsName = ""

	xlsx = xlrd.open_workbook(xlsName)

	for sheet in sheets:
		bookSheet = xlsx.sheet_by_name(sheet)

		getSqlObj = GetSqlObj(bookSheet,"d:/ecif_sql")
		getSqlObj.getSql()
		getSqlObj.writeSql()
		getSqlObj.getDropSql()
		getSqlObj.writeSql(True)
