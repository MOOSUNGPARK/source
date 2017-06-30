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
        self._sub_url = 'http://www.simplyscripts.com/movie-screenplays.html'
        self._download_url = []
        self._set_chrome_driver()

    def _set_chrome_driver(self):
        self.driver = webdriver.Chrome(SubtitleCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _get_sub_url(self):
        self.driver.get(self._sub_url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        body = soup.find_all('p')

        for idx, tag in enumerate(body):
            try:
                url = tag.a['href']
                extension = url.split('.')[-1].lower()
                self._make_save_dir(extension)
                self._download_url.append([url, extension])
            except:
                print('no.' + str(idx), 'url load failed')

    def _sub_downloads(self):
        for idx, urlextension in enumerate(self._download_url):
            try:
                url = urlextension[0]
                extension = urlextension[1]

                if extension in ('html', 'htm'):
                    urlretrieve(url, SubtitleCrawler.__SAVE_FILE_PATH + 'txt\\m{}.txt'.format(idx))
                else:
                    urlretrieve(url, SubtitleCrawler.__SAVE_FILE_PATH + '{0}\\m{1}.{0}'.format(extension, idx))
                if idx / 10 == idx // 10:
                    print('no.' + str(idx), 'subtitle downloading')

            except:
                print('no.' + str(idx), 'subtitle downloading failed')

    def _make_save_dir(self, extension):
        save_path = SubtitleCrawler.__SAVE_FILE_PATH + '{}\\'.format(extension)
        if not os.path.isdir(os.path.split(save_path)[0]) and extension not in ('html', 'htm'):
            os.mkdir(os.path.split(save_path)[0])

    def _pdf_to_txt(self):
        pdf_file_path = SubtitleCrawler.__SAVE_FILE_PATH + 'pdf\\'
        txt_file_path = SubtitleCrawler.__SAVE_FILE_PATH + 'txt\\'
        filenames = os.listdir(pdf_file_path)  # 지정된 폴더 내 파일이름들 불러오기
        full_filenames = []

        for filename in filenames:
            # print(os.path.join(pdf_file_path, filename))
            # print(filename.split('.')[0])
            full_filenames.append(
                [filename.split('.')[0], os.path.join(pdf_file_path, filename)])  # full_filename = 경로+파일이름

        for idx, full_filename in enumerate(full_filenames):
            try:
                rsrcmgr = PDFResourceManager()
                retstr = io.StringIO()
                codec = 'utf-8'
                laparams = LAParams()
                device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                password = ""
                maxpages = 0
                caching = True
                pagenos = set()
                fp = open(full_filename[1], 'rb')

                for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                              password=password,
                                              caching=caching,
                                              check_extractable=True):
                    interpreter.process_page(page)

                text = retstr.getvalue()

                fp.close()
                device.close()
                retstr.close()
                # return text
                ital = open(txt_file_path + '{}.txt'.format(full_filename[0]), 'w', encoding='UTF-8', newline='')
                ital.write(text)
                ital.close()
                if idx / 10 == idx // 10:
                    print('no.' + str(idx), 'pdf converting')

            except:
                print('no.' + str(idx), 'converting failed')

    def play_crawler(self):
        print('crawling start.')
        self._get_sub_url()
        print('crawling complete.')
        print('subtitle downloading.')
        self._sub_downloads()
        print('downloading complete.')
        print('converting pdf to txt')
        self._pdf_to_txt()
        print('converting complete')

if __name__ == '__main__':
    crawler = SubtitleCrawler()
    crawler.play_crawler()