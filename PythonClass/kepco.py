import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import selenium
import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import

########################### INPUT 정보 ###########################
ID = '0422202556'
PW = 'bb772771'
Input_Year = '2015'
Input_Month = '01'
Input_Day = '05'


########################### url 받아오기 ###########################

binary = 'c:\chromedriver/chromedriver.exe'
browser = webdriver.Chrome(binary)
browser.get("https://pccs.kepco.co.kr/iSmart/jsp/cm/login/login.jsp")

########################### 로그인 ###########################

def login(ID,PW):
    id = browser.find_element_by_name("userId")   # 아이디 타이핑 하는 곳
    pw = browser.find_element_by_name("password")   # 비번 타이핑 하는 곳
    time.sleep(3)
    id.send_keys(ID)    # 아이디 입력
    pw.send_keys(PW)  # 비밀번호 입력
    pw.submit()

login(ID,PW)

########################### 클릭해서 찾아가기 ###########################
def click(input_year, input_month, input_day):
    clickloc_year = "//select[@name='year']//option[@value = {}]".format(input_year)
    year =  browser.find_element_by_xpath(clickloc_year)
    year.click()

    clickloc_month = "//select[@name='month']//option[@value = {}]".format(input_month)
    month =  browser.find_element_by_xpath(clickloc_month)
    month.click()

    clickloc_day = "//select[@name='day']//option[@value = {}]".format(input_day)
    day =  browser.find_element_by_xpath(clickloc_day)
    day.click()

    clickloc_image = "//img[@src='/iSmart/images/new/btn_search.gif']"
    img =  browser.find_element_by_xpath(clickloc_image)
    img.click()

click(Input_Year,Input_Month,Input_Day)

html = browser.page_source         # 크롬브라우져에서 현재 불러온 소스 가져옴
soup = BeautifulSoup(html, "lxml") # html 코드를 검색할 수 있도록 설정

print(soup)
# /html/body/form[2]/table[12]/tbody/tr[3]/td/table


# ########################### 그림파일 저장 ###########################
#
# def fetch_list_url():
#     params = []
#     contents = soup.find_all("table", class_="table02")
#     print(contents)
    # for content in contents:
    #     params.append(content)
    #     # except KeyError:
    #     #     params.append(content["data-src"])
    # return params

fetch_list_url()
# def fetch_detail_url():
#     params = fetch_list_url()
#
#     for idx,p in enumerate(params,1):
#         # 다운받을 폴더경로 입력
#         urllib.request.urlretrieve(p, "c:/googleImages/" + str(idx) + ".jpg")
#
# if __name__ == '__main__':
#     # 메인 실행 함수
#     fetch_detail_url()
#
#     # 끝나면 브라우져 닫기
#     browser.quit()

