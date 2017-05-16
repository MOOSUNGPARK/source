from bs4 import BeautifulSoup
#
# with open(r"C:\python\data\abc.html") as aaa:
#     soup = BeautifulSoup(aaa, "lxml")
#
# print (soup.title.text)
# print(soup.find('a'))     #a 태그의 첫번째 요소만 한 줄로 가져오기
# print(soup.find_all('a')) #a 태그의 모든 요소를 한 줄로 가져오기
#
# for link in soup.find_all('a'):
#     print(link)                 #link 주소가 있는 요소를 가져오기
#     '''<a class="cafe1" href="http://cafe.daum.net/oracleoracle" id="link1"><div>다음카페</div></a>'''
#     print(link.get('href'))     #link 주소를 가져오기
#     '''http://cafe.daum.net/oracleoracle'''
# print(soup.get_text())            #텍스트만 가져오기
# print(soup.get_text(strip=True))  #텍스트를 한줄로 가져오기

with open(r"C:\python\data\ee.html") as eee:
    soup = BeautifulSoup(eee, "lxml")

import urllib.request
from bs4 import BeautifulSoup
import re
import os


def fetch_list_url(page):
    list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?c.page={}&hmpMnuId=106&searchKeywordValue=0&bbsId=10059819&searchKeyword=&searchCondition=&searchConditionValue=0&".format(page)
    url = urllib.request.Request(list_url)
    res = urllib.request.urlopen(url).read().decode("utf-8")
    # 위 두가지 작업 이후 url 의 html 문서를 res 변수에 담을 수 있음
    soup_packtpage = BeautifulSoup(res, "html.parser")
    # res가 담긴 html 코드를 BeautifulSoup 모듈로 검색하기 위한 작업
    # '완젼'이라는 텍스트를 검색하기 위해 완젼이라는 한글을 컴파일 함
    b = soup_packtpage.find_all('p', class_='con')
    c = soup_packtpage.find_all('span',class_= 'date')

    for idx,link in enumerate(b,0):
        print(c[idx].text, link.get_text(strip=True))

for cnt in range(1,16):
    fetch_list_url(cnt)
