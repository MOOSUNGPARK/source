# # http://www.kma.go.kr/weather/climate/past_table.jsp?      stn=108    &   yy=2016   &&   obs=07  (지역, 연도, 요소 순이다)
# # 기상청 데이터를 스크롤링 하려면 크게 2가지 기능이 필요하다
# # 1. 지역, 연도, 요소에 해당하는 숫자값을 기상청 html에서 가져오는 기능
# # 2. 3개의 숫자값을 이용해서 상세 url을 완성해서 온도 데이터를 가져오는 기능
#
#
# # 1. 지역, 연도, 요소에 해당하는 숫자값을 기상청 html에서 가져오는 기능
# # -*- coding: utf-8 -*-
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
# import operator
# import time
#
#
# class KMACrawler:
#     FILE_PATH = 'c:\\data\\'  # 기상데이터를 수지발 위치를 지정
#
#     def __init__(self):
#         self.location_list = {}   # 지역의 데이터를 담을 변수
#         self.year_list = {}       # 연도의 데이터를 담을 변수
#         self.factor_list = {}     # 요소의 데이터를 담을 변수
#         self.crawling_list = {}   # 위의 3가지 데이터를 조합해서 원하는 데이터를 담을 변수, (ex. 서울, 2015년, 상대습도)
#         self.data = {}            # 실제 결과 데이터를 담을 딕셔너리 변수 (ex. 3.7, -1.5, 7.0, ....)
#
#         self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
#         # 기상청 온도 확인하는 메인 url
#         self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'
#         # 특정 지역, 요소에 따른 데이터를 조회하는 상세 url
#
#     # 지점, 연도, 요소에 데이터 가져오는 함수
#     def get_kma_data(self):
#         res = urlopen(Request(self.default_url)).read()
#         # 메인 url을 통해서 html코드를 가져오는 부분
#         # >>b'\r\n\r\n\r\n\r\n\r\n\r\n\r\n\t<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" .............
#
#         # print(res)
#
#         soup = BeautifulSoup(res, 'html.parser')
#         location = soup.find('select', id='observation_select1')   # 지역
#         year = soup.find('select', id='observation_select2')       # 연도
#         factor = soup.find('select', id='observation_select3')     # 요소
#         for tag in location.find_all('option'):    # option태그에 해당하는 부분만 가져온다.
#             if tag.text != '--------':       # 구분선은 가져오지 마!
#                 self.location_list[tag['value']] = tag.text
#                 # print(tag['value']) # 277
#                 # print(tag.text)     # 문경(무)
#         for tag in year.find_all('option'):        # option태그에 해당하는 부분만 가져온다.
#             if tag.text != '--------':       # 구분선은 가져오지 마!
#                 self.year_list[tag['value']] = tag.text
#                 # print(tag['value']) # 1961,  key와 value의 값이 같다.
#                 # print(tag.text)     # 1961
#         for tag in factor.find_all('option'):      # option태그에 해당하는 부분만 가져온다.
#             if tag.text != '--------':       # 구분선은 가져오지 마!
#                 self.factor_list[tag['value']] = tag.text
#                 # print(tag['value']) # 06
#                 # print(tag.text)     # 평균풍속
#         # print(self.location _list.items())
#         # print(self.year_list.items())
#         # print(self.factor_list.items())
#         for loc_key, loc_value in self.location_list.items():
#             for year_key, year_value in self.year_list.items():
#                 for fac_key, fac_value in self.factor_list.items():
#                     self.crawling_list[(loc_key, year_key, fac_key)] = (loc_value, year_value, fac_value)
#                     # key와 value를 차곡차곡 넣고 있다. 경우의 수를 다 넣은 것이다.
#         print(self.crawling_list)
#
#
# crawler = KMACrawler()
# crawler.get_kma_data()

##########################################################
##########################################################
##########################################################

# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import operator
import time

class KMACrawler:
    FILE_PATH = 'c:\\data\\'
    def __init__(self):
        self.location_list = {}
        self.year_list = {}
        self.factor_list = {}
        self.crawling_list = {}
        self.data = {}
        self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
        self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'
        self.play_crawling()

    def get_kma_data(self):
        res = urlopen(Request(self.default_url)).read()
        # 메인 url을 통해서 html코드를 가져오는 부분
        # >>b'\r\n\r\n\r\n\r\n\r\n\r\n\r\n\t<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" .............

        # print(res)

        soup = BeautifulSoup(res, 'html.parser')
        location = soup.find('select', id='observation_select1')   # 지역
        year = soup.find('select', id='observation_select2')       # 연도
        factor = soup.find('select', id='observation_select3')     # 요소
        for tag in location.find_all('option'):    # option태그에 해당하는 부분만 가져온다.
            if tag.text != '--------':       # 구분선은 가져오지 마!
                self.location_list[tag['value']] = tag.text
                # print(tag['value']) # 277
                # print(tag.text)     # 문경(무)
        for tag in year.find_all('option'):        # option태그에 해당하는 부분만 가져온다.
            if tag.text != '--------':       # 구분선은 가져오지 마!
                self.year_list[tag['value']] = tag.text
                # print(tag['value']) # 1961,  key와 value의 값이 같다.
                # print(tag.text)     # 1961
        for tag in factor.find_all('option'):      # option태그에 해당하는 부분만 가져온다.
            if tag.text != '--------':       # 구분선은 가져오지 마!
                self.factor_list[tag['value']] = tag.text
                # print(tag['value']) # 06
                # print(tag.text)     # 평균풍속
        # print(self.location _list.items())
        # print(self.year_list.items())
        # print(self.factor_list.items())
        for loc_key, loc_value in self.location_list.items():
            for year_key, year_value in self.year_list.items():
                for fac_key, fac_value in self.factor_list.items():
                    self.crawling_list[(loc_key, year_key, fac_key)] = (loc_value, year_value, fac_value)
                    # key와 value를 차곡차곡 넣고 있다. 경우의 수를 다 넣은 것이다.
        print(self.crawling_list)
        return self.crawling_list

    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        print('크롤링을 위한 데이터를 수집 중입니다...')
        self.crawling_list = self.get_kma_data()
        print('크롤링을 위한 데이터 수집 완료 !!!')
        # print(self.crawling_list.items())
        print('크롤링을 시작합니다...')
        for key, value in sorted(self.crawling_list.items(), key=operator.itemgetter(0)):
            # print('key', key)
            # print('value', value)
            res = urlopen(Request(self.crawled_url.format(key[0], key[1], key[2]))).read()
            # 상세 url 완성해서 기상청 웹서버에 요청한 후 html문서 받아옴
            # print('res', res)
            soup = BeautifulSoup(res, 'html.parser')
            print('현재 키워드 : {}, {}, {}'.format(*value))
            # print('현재 키워드 : {}, {}, {}'.format(value[0], value[1], value[2])) 이런 뜻이다.
            for tr_tag in soup.find('table', class_='table_develop').find('tbody').find_all('tr'):
                # print('tr_tag', tr_tag)
                if self.data.get(value) is None:
                    # print(self.data.get(value))
                    self.data[value] = []   # data딕셔너리의 key에 []를 넣어라
                self.data[value].append([td_tag.text for td_tag in tr_tag.find_all('td')])
                self.data[value].append(['' if td_tag.text=='\xa0' else td_tag.text for td_tag in tr_tag.find_all('td') if td_tag.has_attr('scope') is False])
                #scope="row" 이걸 없애고 싶다!! 즉 '1일', '2일', ...
                # td_tag의 text들을 넣은 list를 계속 담아준다. 즉 ['5일', '21.1', '18.9', ...]
            print (self.data.items())
            print('{}, {}, {} 에 대한 데이터 저장...'.format(*value))
            self.data_to_file()
            self.data.clear()
            print('저장 완료!!!\n\n')
            time.sleep(2)
        print('크롤링 완료 !!!')

    def data_to_file(self):
        with open(KMACrawler.FILE_PATH + "kma_crawled.txt", "a", encoding="utf-8") as file:
            file.write('======================================================\n')
            for key, value in self.data.items():
                file.write('>> ' + key[0] + ', ' + key[1] + ', ' + key[2] + '\n')
                for v in value:
                    file.write(','.join(v) + '\n')
            file.write('======================================================\n\n')
            file.close()

    # def get_kma_data(self):

crawler = KMACrawler()
crawler.play_crawling()



