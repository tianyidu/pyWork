# import tkinter

# top = tkinter.Tk()
# label = tkinter.Label(top,text='hello world')
# label.pack()

# tkinter.mainloop()

from tkinter import *

def resize(ev=None):
	print('Helvetica -%d bold' % (scale.get()))
	label.config(font='Helvetica -%d' % (scale.get()))

top=Tk()
top.geometry('250x150')

label = Label(top,text='hello world',font='Helvetica -12 bold')
label.pack(fill=Y,expand=1)

scale = Scale(top,from_=10,to=40,orient=HORIZONTAL,command=resize)
scale.set(12)
scale.pack(fill=X,expand=1)

quit =  Button(top,text='QUIT',command=top.quit,activeforeground='green',activebackground='red')
quit.pack()

mainloop()