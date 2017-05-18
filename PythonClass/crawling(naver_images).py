import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import

########################### url 받아오기 ###########################

# 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다
# 팬텀 js로 하면 백그라운드로 실행할 수 있음
binary = 'c:\chromedriver/chromedriver.exe'

# 브라우져를 인스턴스화
browser = webdriver.Chrome(binary)

# 네이버의 이미지 검색 url 받아옴(아무것도 안 쳤을때의 url)
browser.get("https://search.naver.com/search.naver?sm=tab_hty.top&where=image&oquery=&ie=utf8&query=")

# 네이버의 이미지 검색에 해당하는 input 창의 id 가 'nx_query' 임(검색창에 해당하는 html코드를 찾아서 elem 사용하도록 설정)
# input창 찾는 방법은 원노트에 있음
# find_elements_by_class_name("") --> 클래스 이름으로 찾을때는 이렇게
elem = browser.find_element_by_id("nx_query")


########################### 검색어 입력 ###########################

# elem 이 input 창과 연결되어 스스로 아이언맨을 검색
elem.send_keys("아이언맨")
# 웹에서의 submit 은 엔터의 역할을 함
elem.submit()

########################### 반복할 횟수 ###########################

# 스크롤을 내리려면 브라우져 이미지 검색결과 부분(바디부분)에 마우스 클릭 한번 하고 End키를 눌러야함
for i in range(1, 5):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(10)                  # END 키 누르고 내려가는데 시간이 걸려서 sleep 해줌

time.sleep(10)                      # 네트워크 느릴까봐 안정성 위해 sleep 해줌
html = browser.page_source         # 크롬브라우져에서 현재 불러온 소스 가져옴
soup = BeautifulSoup(html, "lxml") # html 코드를 검색할 수 있도록 설정


# print(soup)
# print(len(soup))

########################### 그림파일 저장 ###########################

def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="_img")  # 네이버 이미지 url 이 있는 img 태그의 _img 클래스에 가서
    for im in imgList:
        params.append(im["src"])                   # params 리스트에 image url 을 담음
    return params


def fetch_detail_url():
    params = fetch_list_url()
    # print(params)
    for idx,p in enumerate(params,1):
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "c:/naverImages/" + str(idx) + ".jpg")

if __name__ == '__main__':
    # 메인 실행 함수
    fetch_detail_url()

    # 끝나면 브라우져 닫기
    browser.quit()