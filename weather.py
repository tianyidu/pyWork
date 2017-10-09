from suds.client import Client
from suds.xsd.doctor import Import
from suds.xsd.doctor import ImportDoctor


class WeatherHelper:

	url="http://www.webxml.com.cn/WebServices/WeatherWebService.asmx?wsdl"
	tns = "http://WebXml.com.cn/"

	def __init__(self):
		imp = Import("http://www.w3.org/2001/XMLSchema", location="http://www.w3.org/2001/XMLSchema.xsd")
		imp.filter.add(self.tns)
		doctor = ImportDoctor(imp)
		self.client = Client(self.url,doctor=doctor)

	def getSupportAreas(self):
		print(self.client.service.getSupportDataSet())

	def getSupportProvinces(self):
		#print(self.client.service.getSupportProvince())
		return self.client.service.getSupportProvince()

	def getSupportCities(self,ProvinceName):
		#print(self.client.service.getSupportCity(ProvinceName))
		return self.client.service.getSupportCity(ProvinceName)

	def getWeatherByCityName(self,CityName):
		#print(self.client.service.getWeatherbyCityName(CityName))
		return self.client.service.getWeatherbyCityName(CityName)

if __name__ == "__main__":
	#print(WeatherHelper().getSupportProvinces())
	#print(WeatherHelper().getSupportCities(input("输入要查询的省份:")))
	# cityweather = WeatherHelper().getWeatherByCityName(input("输入要查询的城市:"))  
	cityweather = WeatherHelper().getWeatherByCityName("北京")  
	print("*"*30)
	print(cityweather[0][0],cityweather[0][1],cityweather[0][6],cityweather[0][7],cityweather[0][5], end='\n')
	print(cityweather[0][10],cityweather[0][11], sep='\n')
	#print(cityweather)
