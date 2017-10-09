from tkinter import Tk
from time import sleep
from tkinter.messagebox import showwarning
import win32com.client as win32

warn = lambda app:showwarning(app,'Exit?')
RANGE = range(3,8)

def excel():
	app = 'excel'
	xl = win32.gencache.EnsureDispatch('%s.Application' % app)
	sleep(2)
	ss = xl.Workbooks.Add()
	sh = ss.ActiveSheet
	xl.Visible = True
	sleep(2)
	sh.Cells(1,1).Value = 'Python-to-%s Demo' % app
	sleep(1)

	for i in RANGE:
		sh.Cells(i,1).Value = 'Line %d' % i
		sleep(2)
	sh.Cells(i+2,1).Value = 'hthhteth'

	warn(app)
	ss.Close(False)
	xl.Application.Quit()

if __name__ == '__main__':
	Tk().withdraw()
	excel()