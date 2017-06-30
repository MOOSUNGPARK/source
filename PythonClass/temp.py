from selenium import webdriver
from urllib.request import urlretrieve
import os
from bs4 import BeautifulSoup
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


class SubtitleCrawler:
    __CHROME_DRIVER_PATH = 'c:\\chromedriver\\'
    __SAVE_FILE_PATH = 'd:\\data\\'

    def __init__(self):
        self._sub_url = 'https://www.gutenberg.org/wiki/Science_Fiction_(Bookshelf)'
        self._download_url = []
        self._set_chrome_driver()

    def _set_chrome_driver(self):
        self.driver = webdriver.Chrome(SubtitleCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _get_sub_url(self):
        self.driver.get(self._sub_url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        body = soup.find_all('ul')
        self._make_save_dir('ebook')

        for idx, tag in enumerate(body):
            try:
                # print(link.a["href"])   #print(link.find('a')[href])도 같음
                url = tag.a['href']

                bookno = url.split('/')[-1]
                self._download_url.append('http://www.gutenberg.org/cache/epub/{0}/pg{0}.txt'.format(bookno))
                # self._make_save_dir(extension)
                # self._download_url.append([url, extension])
            except:
                print('no.' + str(idx), 'url load failed')

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
        print(os.path.split(save_path)[0])
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