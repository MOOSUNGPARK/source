from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os


class TwitCrawler():

    def __init__(self,keyword,input_since,input_until,count):
        self.crawURL = 'https://twitter.com/search-advanced'
        self.FILE_PATH = 'D:\\twit\\twit.txt'
        self.keyword = str(keyword)
        self.input_since = str(input_since)
        self.input_until = str(input_until)
        self.count = int(count)

    def get_save_path(self):
        save_path = self.FILE_PATH.replace("\\", "/")
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])
        return save_path


    def get_twit_data(self):

        binary = 'D:\chromedriver/chromedriver.exe'
        browser = webdriver.Chrome(binary)
        browser.get(self.crawURL)
        elem = browser.find_element_by_name("ands")
        since = browser.find_element_by_id("since")
        until = browser.find_element_by_id("until")
        #find_elements_by_class_name("")
        elem.send_keys(self.keyword)
        since.send_keys(self.input_since)
        until.send_keys(self.input_until)
        elem.submit()
        for i in range(1,self.count):
            browser.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(5)

        time.sleep(5)
        html = browser.page_source  # 내가 브라우져로 보고있는 소스를 볼려고하는것이다.
                                    # 그런데 그냥 열면 사용자가 end 버튼틀 눌러서 컨트롤
                                    # 한게 반영 안된것이 열린다.
        soup = BeautifulSoup(html,"lxml")
        #print(soup)
        #print(len(soup))
        self.tweet_tag = soup.find_all('div', class_="js-tweet-text-container")
        browser.quit()

    def write_twit_data(self):
        file = open(self.get_save_path(), 'w', encoding='utf-8')
        self.get_twit_data()
        for i in self.tweet_tag:
            tweet_text = i.get_text(strip=True)
            print(tweet_text)
            file.write(tweet_text)
        file.close()

twit = TwitCrawler('삼성전자','2015/01/01','2015/12/31',10)
twit.write_twit_data()



#####################################################################################


from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import operator


class TwitterCrawler:
    CHROME_DRIVER_PATH = 'D:\\chromedriver\\'
    FILE_PATH = 'D:\\twitter_data\\'

    def __init__(self, search_word, start_date, end_date, routine):
        self.search_word = search_word
        self.start_date = start_date
        self.end_date = end_date
        self.routine = routine
        self.data = {}
        self.url_form = 'https://twitter.com/search?l=&q=' + search_word + '%20since%3A' + start_date + '%20until%3A' + end_date + '&src=typd&lang=ko'
        self.set_chrome_driver()
        self.play_crawling()

    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(TwitterCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')

    def page_scroll_down(self):
        for i in range(0, self.routine):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(5)

    def data_to_file(self):
        with open(TwitterCrawler.FILE_PATH + self.search_word + ".txt", "w",
                  encoding="utf-8") as file:  # PATH\키워드.txt를 쓰기가능한 유니코드파일로 열면서
            print('데이터를 저장하는 중입니다.')  # 프린트하며
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):  # data 딕셔너리에, 정렬하여 넣겠다.
                # data.items() 에 key 와 value 가 들어있고 그리고 0 번째 요소로 정령하겠다.
                file.write("==" * 30 + "\n")
                file.write(key + "\n")
                file.write(value + "\n")
                file.write("==" * 30 + "\n")

                file.write(value + '\n')  # 밸류값을 파일에 작성한다.
            file.close()  # 파일종료
            print('데이터 저장이 완료되었습니다.')

    def play_crawling(self, ):
        try:
            self.driver.get(self.url_form)
            self.page_scroll_down()
            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            content_find = soup.find_all("div", class_="content")  # len(18)
            for tag in content_find:
                usertime = tag.find('small', class_='time').find('a')['title']  # 타이틀 자체가 값이라서 get_text 안함
                text = tag.find('p')
                # print(text)
                if usertime is not None and text is not None:
                    self.data[usertime] = text.get_text(strip=True)
            self.data_to_file()
            self.driver.quit()
            print(self.data)
        except:
            print('정상 종료 되었습니다.')


crawler = TwitterCrawler('문재인', '2015-02-01', '2017-05-01', 100)
crawler.play_crawling()

