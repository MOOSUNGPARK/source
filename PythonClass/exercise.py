from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.request import urlretrieve
import os
from bs4 import BeautifulSoup


class SubtitleCrawler:
    __CHROME_DRIVER_PATH = 'c:\\chromedriver\\'
    __SAVE_FILE_PATH = 'c:\\data\\'

    def __init__(self):
        self._sub_url = 'http://www.fullbooks.com/'
        self._download_url = []
        self._set_chrome_driver()

    def _set_chrome_driver(self):
        self.options = Options()
        self.options.add_extension(self.__CHROME_DRIVER_PATH + 'zenmate.crx')
        self.driver = webdriver.Chrome(executable_path= SubtitleCrawler.__CHROME_DRIVER_PATH + 'chromedriver',
                                      chrome_options=self.options )
        # self.driver = webdriver.Chrome(executable_path= SubtitleCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _get_sub_url(self, page):
        if page <= 54 :
            self.driver.get(self._sub_url+'idx{}.html'.format(page))
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            body = soup.find_all('li')
            self._make_save_dir('fullbook')

            for idx, tag in enumerate(body):
                try :
                    url = tag.find_all('a')

                    for tag2 in url:
                        url2 = tag2.get('href')
                        self.driver.get(self._sub_url + url2)
                        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                        body2 = soup.find_all('li')
                        if body2 :
                            for tag3 in body2:
                                url3 = tag3.a['href']
                                self._download_url.append(self._sub_url + url3)
                        else :
                            self._download_url.append(self._sub_url + url2)
                except:
                    continue
            print(page,'-----------------',self._download_url)
            # self._sub_downloads(page)
            self._download_url = []
            self._get_sub_url(page+1)
        else :
            return

    def _sub_downloads(self, page):
        for idx, urlextension in enumerate(self._download_url):
            try:
                url = urlextension
                print(url)
                urlretrieve(url, SubtitleCrawler.__SAVE_FILE_PATH + 'fullbook\\f{0}_{1}.txt'.format(page,idx))
                if idx / 10 == idx // 10:
                    print('no.' + str(page) + '_' + str(idx), 'subtitle downloading')
            except:
                print('no.' + str(page) + '_' + str(idx), 'subtitle downloading failed')

    def _make_save_dir(self, extension):
        save_path = SubtitleCrawler.__SAVE_FILE_PATH + '{}\\'.format(extension)
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])

    def play_crawler(self):
        print('crawling start.')
        self._get_sub_url(9)
        print('crawling complete.')
        print('subtitle downloading.')
        # self._sub_downloads()
        print('downloading complete.')

if __name__ == '__main__':
    crawler = SubtitleCrawler()
    crawler.play_crawler()