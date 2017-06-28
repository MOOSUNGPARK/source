from selenium import webdriver
from urllib.request import urlretrieve
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
from PIL import Image


binary = 'c:\chromedriver/chromedriver.exe'
browser = webdriver.Chrome(binary)
browser.get("http://www.simplyscripts.com/tv_all.html")

html = browser.page_source
soup = BeautifulSoup(html, "html.parser")
content_find = soup.find_all("td", class_="list-subject")  # len(18)
for idx, tag in enumerate(content_find,1):
    # print(tag.a['href'])
    splittag = tag.a['href'].split('&')
    # print(splittag)
    downloadtag = 'http://cineaste.co.kr/bbs/download.php?bo_table=psd_caption&' + splittag[1] + '&no=0&ds=1&js=on&' + splittag[2] + '&' + splittag[3]
    urlretrieve(downloadtag,'d:/data/{}.txt'.format(idx))
    print(downloadtag)