from selenium import webdriver
import time
import os
"""
For IE 11 only, you will need to set a registry entry on the target computer so that the driver can maintain a connection to the instance of Internet Explorer it creates. （这段告诉你需要修改注册表。）

For 32-bit Windows installations, the key you must examine in the registry editor is HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE. （32bit Windows看这里。）

For 64-bit Windows installations, the key is HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE. Please note that the FEATURE_BFCACHE subkey may or may not be present, and should be created if it is not present. Important: Inside this key, create a DWORD value named iexplore.exe with the value of 0.（64bit Windows看这里。）

翻译过来的意思即，修改你的注册表（Run->regedit->Enter），路径如下：

HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BFCACHE

如果FeatureControl下没有FEATURE_BFCACHE，就以FEATURE_BFCACHE为名new一个key！并在其下创建一个DWORD，取名为：iexplore.exe，value为0。
"""


def openLink(driver,linkName,parentLink,loopCnt=1):
	try:
		ele = driver.find_element_by_link_text(linkName)
	except:
		driver.find_element_by_link_text(parentLink).click()
		ele = driver.find_element_by_link_text(linkName)
	for i in range(loopCnt):
		ele.click()
		time.sleep(1)

def openReport(driver,report_l1,report_l2,report_l3):
	
	#print(driver.find_element_by_id("main").text())

	openLink(driver, report_l1,report_l1)
	openLink(driver, report_l2,report_l1)
	openLink(driver, report_l3,report_l2,2)

	driver.get_screenshot_as_file(os.path.join("d:/",report_l3+".jpg"))

if __name__=="__main__": 
	#report_l1 = ["报表1","报表2","报表3"]
	#report_l2 = ["情况统计表","出险频率报表_出险件数","业务情况统计表"]
	#report_l3 = ["情况统计表","险频率报表","情况统计表"]
	report_l1 = ["报表"]
	report_l2 = ["业务情况统计表"]
	report_l3 = ["情况统计表"]

	url = "http://10.8.20.110:7000/cognos"
	driver = webdriver.Ie()
	driver.get(url)

	driver.find_element_by_name("usercode").send_keys("") #用户名
	driver.find_element_by_name("password").send_keys("") #密码

	driver.find_element_by_id("").submit()  #form表单id
	time.sleep(1)
	leftFrame = driver.find_element_by_id("leftFrame")
	driver.switch_to.frame(leftFrame)

	for i in range(len(report_l1)):
		openReport(driver,report_l1[i], report_l2[i], report_l3[i])

	print("driver closing")
	driver.quit()
