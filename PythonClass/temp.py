'''

1. 게시물 url-> 
2. smi/ rar/ zip 나눠서 url 저장하고 각 폴더에 한꺼번에 크롤링해서 다운받기 

'''


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import urllib.request
from PIL import Image

class DramaCrawler:
    __CHROME_DRIVER_PATH = 'c:\\'
    __IMAGE_PATH = 'c:\\python\\data\\drama'
    __SEARCH_KEYWORD = ['개']

    def __init__(self):
        self.__image_urls = []
        self._google_image_url = 'https://www.google.com/imghp?hl=ko'
        self._drama_sub_url = 'http://cineaste.co.kr/bbs/board.php?bo_table=psd_dramacap&page='
        self._set_chrome_driver()


    def _set_chrome_driver(self):
        self.driver = webdriver.Chrome(DramaCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _extract_image_url(self, images):
        number = 1
        for image in images:
            try:
                image_src = image['src']
                if image_src is not None:
                    self.__image_urls.append((number, image_src))
                    number += 1
            except KeyError:
                print(image['name'] + ', src 속성이 존재하지 않습니다.')

    def _get_image_crawling(self, keyword):
        self.driver.get(self._google_image_url)
        self.driver.find_element_by_id("lst-ib").clear()
        self.driver.find_element_by_id("lst-ib").send_keys(keyword)
        self.driver.find_element_by_id("lst-ib").submit()
        time.sleep(3)

        before_img_cnt = 0
        clicked = False

        while True:
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            images = soup.find_all('img', class_='rg_ic rg_i')
            after_img_cnt = len(images)

            if before_img_cnt == after_img_cnt:
                if soup.find('input', id='smb') and clicked is False:
                    self.driver.find_element_by_id('smb').click()
                    clicked = True
                    time.sleep(3)
                else:
                    self._extract_image_url(images)
                    self.driver.quit()
                    break
            else:
                before_img_cnt = after_img_cnt
                time.sleep(3)

    def _image_downloads(self):
        for name, url in self.__image_urls:
            urllib.request.urlretrieve(url, DramaCrawler.__IMAGE_PATH + str(name) + '.jpg')

    def _image_to_thumbnail(self):
        for name, _ in self.__image_urls:
            try:
                im = Image.open(DramaCrawler.__IMAGE_PATH + str(name) + '.jpg')
                im.thumbnail((128, 128), Image.ANTIALIAS)
                im.save(DramaCrawler.__IMAGE_PATH + str(name) + '_thumbnail.jpg')
            except:
                print(str(name) + '.jpg, file is not exists.')

    def play_crawler(self):
        for keyword in DramaCrawler.__SEARCH_KEYWORD:
            print('crawling start.')
            self._get_image_crawling(keyword)
            print('crawling complete.')
            print('image count : ' + str(len(self.__image_urls)))

            print('image downloading.')
            self._image_downloads()
            print('image downloading complete.')

            print('image to thumbnail start.')
            self._image_to_thumbnail()
            print('image to thumbnail end.')

            self.__image_urls.clear()

crawler = DramaCrawler()
crawler.play_crawler()