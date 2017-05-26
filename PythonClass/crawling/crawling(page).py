import urllib.request           # 웹브라우저에서 html문서를 얻어오기 위해 통신하기 위한 모듈
from bs4 import BeautifulSoup   # html문서 검색 모듈
import os


def get_save_path():
    save_path = input('Enter the file name and file location : ')
    save_path = save_path.replace('\\', '/')

    # 폴더가 없으면 폴더를 만드는 작업
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])

    return save_path


def fetch_list_url(page):
    params = []

    for cnt in range(page):
        list_url = "http://search.joins.com/JoongangNews?page={}&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New&SearchCategoryType=JoongangNews".format(cnt)
                    ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.

        url = urllib.request.Request(list_url)                          # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
                                                                        # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
        '''
        참고>>  문자를 담는 set:
                1. 아스키코드 : 영문
                2. 유니코드 : 한글, 중국어
        '''
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정

        for link in soup.find_all('span',class_='thumb'):   #p에는 이미지가 있는 뉴스만 있다. 이미지 없는 뉴스도 가져오기 위해서 dt사용
            try:
                # print(link.a["href"])   #print(link.find('a')[href])도 같음
                params.append(link.a["href"])
            except:                        #<dt></dt>처럼 비어있는 곳이 있기 때문에 try except를 사용해야 한다.
                continue
    # print(params)
    return params

def fetch_list_url2(page):
    params2 = fetch_list_url(page)
    f = open(get_save_path(),'w',encoding='utf-8')

    for i in params2:
        list_url = "{}".format(i)
        ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.

        url = urllib.request.Request(list_url)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")  # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup2 = soup.find_all('div', class_='article_body fs1 mg')[0].get_text(strip=True, separator='\n')
        f.write(soup2 + '\n')
    f.close()

fetch_list_url2(5)


