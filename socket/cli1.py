from socket import *

HOST = 'localhost'
PORT = 3999
BUFSIZE = 1024
ADDR = (HOST,PORT)

tcpCliSock = socket(AF_INET,SOCK_STREAM)
tcpCliSock.connect(ADDR)

while True:
	data = input('请输入要发送的数据：')
	if not data:
		break
	print(data)
	tcpCliSock.send(data.encode())
	data = tcpCliSock.recv(BUFSIZE).decode()
	if not data:
		break
	print(data)
tcpCliSock.close()
