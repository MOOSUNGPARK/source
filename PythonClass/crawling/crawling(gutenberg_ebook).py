from selenium import webdriver
from urllib.request import urlretrieve
import os
from bs4 import BeautifulSoup


class SubtitleCrawler:
    __CHROME_DRIVER_PATH = 'c:\\chromedriver\\'
    __SAVE_FILE_PATH = 'c:\\data\\'

    def __init__(self):
        self._sub_url = 'https://www.gutenberg.org/wiki/Category:Fiction_Bookshelf'
        self._download_url = []
        self._set_chrome_driver()

    def _set_chrome_driver(self):
        self.driver = webdriver.Chrome(SubtitleCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _get_sub_url(self):
        self.driver.get(self._sub_url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        body = soup.find_all('li')
        self._make_save_dir('ebook')

        for idx, tag in enumerate(body):
            try:
                url = tag.find_all('a')

                for tag2 in url:
                    url2 = tag2.get('href')
                    category = url2.split('/')[-1]

                    self.driver.get('https://www.gutenberg.org/wiki/'+category)
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    body2 = soup.find_all('li')

                    for idx3, tag3 in enumerate(body2):
                        url3 = tag3.a['href']
                        bookno = url3.split('/')[-1]
                        if bookno.isdigit():
                            self._download_url.append('http://www.gutenberg.org/cache/epub/{0}/pg{0}.txt'.format(bookno))

            except:
                continue
                # print('no.' + str(idx), 'url load failed')

    def _sub_downloads(self):
        for idx, urlextension in enumerate(self._download_url):
            try:
                url = urlextension
                print(url)
                urlretrieve(url, SubtitleCrawler.__SAVE_FILE_PATH + 'ebook\\e{0}.txt'.format(idx))
                if idx / 10 == idx // 10:
                    print('no.' + str(idx), 'subtitle downloading')
            except:
                print('no.' + str(idx), 'subtitle downloading failed')

    def _make_save_dir(self, extension):
        save_path = SubtitleCrawler.__SAVE_FILE_PATH + '{}\\'.format(extension)
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])

    def play_crawler(self):
        print('crawling start.')
        self._get_sub_url()
        print('crawling complete.')
        print('subtitle downloading.')
        self._sub_downloads()
        print('downloading complete.')

if __name__ == '__main__':
    crawler = SubtitleCrawler()
    crawler.play_crawler()