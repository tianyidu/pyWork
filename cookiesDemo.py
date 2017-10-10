import urllib.request
import http.cookiejar

#url = "http://10.8.20.110:7000/cognos/"
url = "http://www.baidu.com"
param = {"username":"","password":""} #用户名密码
postdata = urllib.parse.urlencode(param).encode("utf-8")

req = urllib.request.Request(url,postdata)
cookies_file = http.cookiejar.MozillaCookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies_file))
r = opener.open(req,postdata)

for co in cookies_file:
	print("cookies",co)
