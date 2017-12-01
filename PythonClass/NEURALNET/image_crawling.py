from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import urllib.request
from PIL import Image
import os
import re
import matplotlib.image as mimage
# import math
import numpy as np


class GoogleImageCrawler:
    __CHROME_DRIVER_PATH = 'C:\\python\\software\\chrome_driver\\'
    # __IMAGE_PATH = ['C:\\python\\data\\face\\happy\\', 'C:\\python\\data\\face\\argry\\', 'C:\\python\\data\\face\\sad\\', 'C:\\python\\data\\face\\surprised\\']
    __IMAGE_PATH = ['C:\\python\\data\\face\\happy\\', 'C:\\python\\data\\face\\argry\\']
    __DATA_PATH = 'C:\\python\\data\\face\\'

    __SEARCH_HAPPY = ['happy human face', '행복한 사람 얼굴']
    __SEARCH_ANGRY = ['angry human face', '화난 사람 얼굴']
    __SEARCH_SAD = ['sad human face', '슬픈 사람 얼굴']
    __SEARCH_SURPRISED = ['surprised human face', '놀란 사람 얼굴']
    # __SEARCH_KEYWORD = [__SEARCH_HAPPY, __SEARCH_ANGRY, __SEARCH_SAD, __SEARCH_SURPRISED]
    __SEARCH_KEYWORD = [__SEARCH_HAPPY, __SEARCH_ANGRY]


    def __init__(self):
        self.__image_urls = []  # 이미지를 다운받을 URL 주소.
        self.__image_data = []  # 이미지가 Gray Scale 로 변환된 데이터.
        self.__number = 1  # 이미지 번호.
        self.__keyword_cnt = len(GoogleImageCrawler.__SEARCH_KEYWORD)  # 검색 종류 개수.
        self._google_image_url = 'https://www.google.com/imghp?hl=ko'
        self.__rgb_cnt = 0
        # self._set_chrome_driver()

    def _set_chrome_driver(self):
        '''
            chrome driver 설정하는 함수.
        '''
        self.driver = webdriver.Chrome(GoogleImageCrawler.__CHROME_DRIVER_PATH + 'chromedriver')


    def _extract_image_url(self, images):
        '''
            html 파일로부터 Image URL 정보를 추출하는 함수.
        '''
        for image in images:
            try:
                image_src = image['src']
                if image_src is not None:
                    self.__image_urls.append((self.__number, image_src))
                    self.__number += 1
            except KeyError:
                print(image['name'] + ', src 속성이 존재하지 않습니다.')

    def _get_image_crawling(self, keyword):
        '''
            특정 키워드에 대한 이미지 검색 후 검색된 이미지들의 URL 주소를 수집하는 함수.
        '''
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
                    try:
                        self.driver.find_element_by_id('smb').click()
                        clicked = True
                        time.sleep(3)
                    except:
                        clicked = True
                        time.sleep(3)
                else:
                    self._extract_image_url(images)
                    break
            else:
                before_img_cnt = after_img_cnt
                time.sleep(3)

    def _image_downloads(self):
        '''
            수집된 이미지 경로를 가지고 직접 이미지를 다운로드하는 함수.
        '''
        for name, url in self.__image_urls:
            urllib.request.urlretrieve(url, GoogleImageCrawler.__IMAGE_PATH[self.__curr_index] + str(name) + '.jpg')

    def _image_to_thumbnail(self):
        '''
            기존 원본 이미지를 특정 사이즈 형식으로 Thumbnail 을 수행하는 함수.
            이미지가 저장된 폴더로부터 이미지를 로드 후 썸네일 이미지 생성.
        '''
        size = (128, 128)
        for index in range(self.__keyword_cnt):
            for file in [filename for filename in os.listdir(GoogleImageCrawler.__IMAGE_PATH[index]) if
                         re.search('[0-9]+\.(jpg|jpeg|png)', filename) is not None]:
                try:
                    print(file)
                    filename, ext = os.path.splitext(file)

                    new_img = Image.new("RGB", (128, 128), "white")
                    im = Image.open(GoogleImageCrawler.__IMAGE_PATH[index] + str(file))
                    im.thumbnail(size, Image.ANTIALIAS)
                    load_img = im.load()
                    load_newimg = new_img.load()
                    i_offset = (128 - im.size[0]) / 2
                    j_offset = (128 - im.size[1]) / 2

                    for i in range(0, im.size[0]):
                        for j in range(0, im.size[1]):
                            load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                    if ext.lower() in ('.jpeg', '.jpg'):
                        new_img.save(GoogleImageCrawler.__IMAGE_PATH[index] + str(filename) + '_128x128.jpeg')
                    elif ext.lower() == '.png':
                        new_img.save(GoogleImageCrawler.__IMAGE_PATH[index] + str(filename) + '_128x128.png')
                except Exception as e:
                    print(str(file), e)

    def _rgb2gray(self, rgb):
        '''
            YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
             - Y = Red*0.2126 + Green*0.7152 + Blue*0.0722
            YPbPr : 아날로그 시스템을 위한 표현방법.
             - Y : Red*0.299 + Green*0.587 + Blue*0.114
            실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
        '''
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        return np.array(gray).astype('int32')

    def _extract_rgb_from_image(self):
        '''
            크롤링 된 이미지 파일들을 읽어들여 Gray Scale 로 변환하는 함수.
        '''
        for index in range(self.__keyword_cnt):
            for name in [filename for filename in os.listdir(GoogleImageCrawler.__IMAGE_PATH[index]) if
                         re.search('[0-9]+\_128x128.+', filename) is not None]:
                try:
                    img = mimage.imread(GoogleImageCrawler.__IMAGE_PATH[index] + str(name))
                    gray = self._rgb2gray(img)
                    self.__image_data.append([gray, index, name])
                except OSError as e:
                    print(str(name) + ', 이미지를 식별할 수 없습니다.', e)
                    continue

                # self.__rgb_cnt += 1
                # if self.__rgb_cnt % 1000 == 0:
                #     self._data_to_file(index)
                #     self.__image_data.clear()
            print(index)
            self._data_to_file(index)
            self.__image_data.clear()
            self.__rgb_cnt = 0

    def _data_to_file(self, index):
        '''
            Gray Scale 로 변환된 이미지 정보를 파일로 기록하는 함수.
        '''
        print('데이터를 저장하는 중입니다.')
        for data in self.__image_data:
            x_shape, y_shape = data[0].shape
            temp_data = ''
            temp_data2 = ''
            temp_data3 = ''
            temp_data4 = ''
            temp_data5 = ''
            temp_data6 = ''
            temp_data7 = ''
            temp_data8 = ''

            for x in range(1, x_shape - 1):
                for y in range(1, y_shape - 1):
                    if x == 1 and y == 1:
                        temp_data += str(data[0][x][y])
                        temp_data2 += str(data[0][x][y] * 0.9)
                        temp_data3 += str(data[0][x][y] * 0.8)
                        temp_data4 += str(data[0][x][y] * 0.7)
                        temp_data5 += str(data[0][127-x][y])
                        temp_data6 += str(data[0][127-x][y] * 0.9)
                        temp_data7 += str(data[0][127-x][y] * 0.8)
                        temp_data8 += str(data[0][127-x][y] * 0.7)

                    else:
                        temp_data += ',' + str(data[0][x][y])
                        temp_data2 += ',' + str(data[0][x][y] * 0.9)
                        temp_data3 += ',' + str(data[0][x][y] * 0.8)
                        temp_data4 += ',' + str(data[0][x][y] * 0.7)
                        temp_data5 += ',' + str(data[0][127-x][y])
                        temp_data6 += ',' + str(data[0][127-x][y] * 0.9)
                        temp_data7 += ',' + str(data[0][127-x][y] * 0.8)
                        temp_data8 += ',' + str(data[0][127-x][y] * 0.7)

            temp_data += ',' + str(data[1])
            temp_data2 += ',' + str(data[1])
            temp_data3 += ',' + str(data[1])
            temp_data4 += ',' + str(data[1])
            temp_data5 += ',' + str(data[1])
            temp_data6 += ',' + str(data[1])
            temp_data7 += ',' + str(data[1])
            temp_data8 += ',' + str(data[1])

            with open(GoogleImageCrawler.__DATA_PATH + 'image_data_' + str(index) + '.csv', 'a', encoding='utf-8') as f:
                # math.ceil(self.__rgb_cnt / 1000)
                f.write(temp_data + '\n' + temp_data2 + '\n' + temp_data3 + '\n' + temp_data4 + '\n' + temp_data5 + '\n' +
                        temp_data6 + '\n' + temp_data7 + '\n' + temp_data8 + '\n')
        print('데이터 저장이 완료되었습니다.')

    def play_crawler(self):
        '''
            이미지 크롤링에 필요한 함수들을 수행하는 함수.
        '''
        self._set_chrome_driver()
        # for index in range(0, self.__keyword_cnt):
        #     self.__curr_index = index
        #     for keyword in GoogleImageCrawler.__SEARCH_KEYWORD[self.__curr_index]:
        #         print('crawling start.')
        #         self._get_image_crawling(keyword)
        #         print('crawling complete.')
        #         print('image count : ' + str(len(self.__image_urls)))
        #
        #         print('image downloading.')
        #         self._image_downloads()
        #         print('image downloading complete.')
        #
        #         self.__image_urls.clear()
        #
        # print('image to thumbnail start.')
        # self._image_to_thumbnail()
        # print('image to thumbnail end.')

        print('rgb to gray start.')
        self._extract_rgb_from_image()
        print('rgb to gray end.')

        self.driver.quit()


crawler = GoogleImageCrawler()
crawler.play_crawler()