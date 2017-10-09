from PIL import Image

class PointInfo:
	def __init__(self,x,y,pixel):
		self.__x = x
		self.__y = y
		self.__pixel = pixel

	def getX(self):
		return self.__x

	def getY(self):
		return self.__y

	def getPixel(self):
		return self.__pixel

def judgeData(imdata,x,y,cnt=0):
	global width
	global height
	if imdata[x][y]==0:
		#print(x,y,cnt)
		if cnt>=width/2 or x<=3 or x>=width-3 or y>=height-3 or y<=3:
			return 1
		cnt+=1
		if imdata[x+1][y+1]==0:
			return judgeData(imdata, x+1, y+1, cnt)
		elif imdata[x+1][y]==0:
			return judgeData(imdata, x+1, y, cnt)
		elif imdata[x+1][y-1]==0:
			return judgeData(imdata, x+1, y-1, cnt)
		else:
			return 0
	else:
		return 0


im = Image.open("C:/Users/dingxian/Pictures/verify.jpg")
im = im.convert("L")
threshold = 140
table = []
for i in range(256):
	if i<threshold:
		table.append(0)
	else:
		table.append(1)

im = im.point(table,'1')
width,height = im.size

imdata = [[] for i in range(width)]

for w in range(width):
	for h in range(height):
		imdata[w].append(im.getpixel((w,h)))

for w in range(width):
	for h in range(height):
		#print("begin judge")
		if judgeData(imdata, w, h)==1:
			#print("judge result ok")
			#if imdata[w][h-1]==1:
				#print("judge result change")
			imdata[w][h]=1

for w in range(width):
	#print(imdata[w])
	for h in range(height):
		im.putpixel((w,h),imdata[w][h])
		#pass

im.show()

