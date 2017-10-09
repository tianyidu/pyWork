import urllib.request
from xml.dom import minidom


#获得html返回报文
def getHtml(urlsource):
	hd=urllib.request.urlopen(urlsource)
	xmldoc=hd.read()
	print(xmldoc)

#使用cElementTree解析xml文件
def parseXml(xmlsource):
	try :
		import xml.etree.cElementTree as ET
	except ImportError:
		import xml.etree.ElementTree as ET
	import sys

	try:
		tree = ET.parse(xmlsource)
		root = tree.getroot()
	except Exception as e:
		raise
	else:
		pass
	finally:
		pass

	print(root.tag,"-----",root.attrib)
	for child in root:
		print(child.tag,"------",child.attrib)
	print(root[0][1].text)

	print("*"*20)

	for country in root.findall("country"):
		print(country.find("year").text)
		print(country.get("name"))

if __name__ == "__main__":
	#getHtml("http://localhost:8088/SyncReport")
	parseXml("D:/wing/wing/PWORKSPACE/test/country.xml")