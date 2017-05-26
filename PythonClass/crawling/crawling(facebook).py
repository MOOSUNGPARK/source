
#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime
import operator
class FacebookCrawler:
    # FILE_PATH = 'D:\\02.Python\\facebook_data\\'
    # CHROME_DRIVER_PATH = 'D:\\02.Python\\'
    FILE_PATH = 'c:\\facebook_data\\'
    CHROME_DRIVER_PATH = 'c:\\chromedriver\\'
    def __init__(self, searchKeyword, startMonth, endMonth, scroll_down_cnt):
        self.searchKeyword = searchKeyword
        self.startMonth = startMonth
        self.endMonth = endMonth
        self.scroll_down_cnt = scroll_down_cnt
        self.data = {}
        self.url = 'https://www.facebook.com/search/str/' + searchKeyword + '/keywords_top?filters_rp_creation_time=%7B"start_month%22%3A"' + startMonth + '"%2C"end_month"%3A"' + endMonth + '"%7D'
        self.set_chrome_driver() # 크롬 드라이브 위치 지정하는 함수 실행
        self.play_crawling()     # 스크롤링을 하는 함수 실행
    # chrome driver 생성 후 chrome 창 크기 설정하는 함수
    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(FacebookCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')
        # self.driver.set_window_size(1024, 768) # 크롬창 설정
    # facebook 홈페이지로 이동 후 email, password 를 입력하고 submit 보내는 함수. (로그인)
    def facebook_login(self):
        self.driver.get("https://www.facebook.com/")  # 페이스북으로 이동
        self.driver.find_element_by_id("email").clear()   # id 입력창을 clear
        self.driver.find_element_by_id("email").send_keys("")   # 아이디
        self.driver.find_element_by_id("pass").clear()    # 비번 창 clear
        self.driver.find_element_by_id("pass").send_keys("") # 비번 입력
        self.driver.find_element_by_id("pass").submit()  # 엔터
        time.sleep(5)  # 로그인하는 5초 간 쉼
        self.driver.get(self.url)  # 페이스북 검색 페이지로 이동
    # facebook page scroll down 하는 함수
    def page_scroll_down(self):
        for i in range(1, self.scroll_down_cnt):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(3)
    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self):
        with open(FacebookCrawler.FILE_PATH + self.searchKeyword + ".txt", "w", encoding="utf-8") as file:
            print('데이터를 저장하는 중입니다.')
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):
                # data.items() 에 key 와 value 가 들어있고 그리고 0 번째 요소(key)로 정렬하겠다.
                file.write(str(datetime.fromtimestamp(key)) + ' : ' + value + '\n')
            file.close()
            print('데이터 저장이 완료되었습니다.')
    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        try:
            self.facebook_login()                      # 페이스북 로그인
            time.sleep(5)                              # 잠깐 5초 로그인
            self.page_scroll_down()                    # 페이스북 스크롤 다운 실행
            html = self.driver.page_source  # 스크롤 다운한 현재까지의 문서를 html 변수에 담아서
            soup = BeautifulSoup(html, "html.parser")  # beautiful soup 로 검색할 수 있도록 설정
            #  . 이 클래스 # 이 id  붙이면 and 떨어뜨리면 or 조건
                # .fbUserContent      ._5pcr
                #   (클래스 이름)       (클래스 이름)
                # .fbUserContent._5pcr (클래스 and 클래스)
                # .fbUserContent      ._5pcr (클래스 or 클래스)
            for tag in soup.select('.fbUserContent._5pcr'):
                usertime = tag.find('abbr', class_='_5ptz')  # 게시한 날짜와 시간
                content = tag.find('div', class_='_5pbx userContent').find('p')   # 게시글
                if usertime is not None and content is not None:
                    self.data[int(usertime['data-utime'])] = content.get_text(strip=True)
                    # data딕셔너리[key]                      = value
            self.data_to_file()  # data_to_file() 함수를 실행해서 data 딕셔너리의 내용을 os의 파일로 생성
            self.driver.quit()
        except:
            print('정상 종료 되었습니다.')
crawler = FacebookCrawler('멍빼', '2012-02', '2017-05', 100)
crawler.play_crawling()
