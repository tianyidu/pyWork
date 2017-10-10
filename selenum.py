from selenium import webdriver
import os
import time
#D:\ProgramFiles\operadriver_win64

url = ""  #访问地址

driver = webdriver.Opera(executable_path="D:\\ProgramFiles\\operadriver_win64\\operadriver.exe")
driver.get(url)

ele_user = driver.find_element_by_name("usercode")
ele_pass = driver.find_element_by_name("password")

ele_user.send_keys("admin")
ele_pass.send_keys("admin")
loginBtn = driver.find_element_by_xpath("//form/dl[3]/dd/input")
loginBtn.click()

time.sleep(1)

ele_myfolder = driver.find_element_by_id("ztree_5_span")
ele_myfolder.click()
time.sleep(1)
ele_download = driver.find_element_by_id("ztree_7_span")
ele_download.click()
time.sleep(1)
driver.switch_to_frame("main")
ele_downreport = driver.find_element_by_xpath('//*[@id="list"]/tbody/tr[2]/td[7]/a')
#print(ele_downreport.get_attribute("href"))
ele_downreport.click()


driver.get_screenshot_as_file("d:/test.jpg")
driver.close()
