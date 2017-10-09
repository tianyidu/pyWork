import pymysql.cursors

print("connect info :")
connect = pymysql.Connect(
	host="127.0.0.1",
	port=3306,
	user="root",
	passwd="",
	db="test"
)

cursor = connect.cursor()

def insertDB():
	
	sql = "insert into t1_user values(%s,%s,%s,%s)"
	para = ("02","admin","admin123","00000000")
	cursor.execute(sql , para)
	connect.commit()
	print("insert %d ok" % cursor.rowcount)

def dropDB(tablename):
	sql = "drop table if exists %s" % tablename
	print(sql)
	cursor.execute(sql)

def createDB():
	sql = "create table t1_user(userid varchar(30),username varchar(60),password varchar(20),comcode varchar(10))"
	cursor.execute(sql)

def updateDB(username):
	sql = "update t1_user set username=%s where userid=%s"
	para = (username,"01")
	cursor.execute(sql ,para)
	connect.commit()
	print("update %d ok" % cursor.rowcount)

def selectDB(userid):
	sql = "select userid,username,password from t1_user where userid=%s"
	para = (userid)
	cursor.execute(sql , para)
	for row in cursor.fetchall():
		print("userid: %s ,username : %s ,passwd :%s" % row)
	print("select %d count" % cursor.rowcount)

def deleteDB():
	sql = "delete from t1_user  where userid=%s"
	para = ("01")
	cursor.execute(sql , para)
	connect.commit()
	print("delete %d ok " % cursor.rowcount)

def close():
	cursor.close()
	connect.close()

if __name__ == "__main__":
	dropDB("t1_user")
	createDB()
	insertDB()
	selectDB("01")
	updateDB("admin01")
	selectDB("0 or 1=1")
	deleteDB()
	selectDB("02")
	close()