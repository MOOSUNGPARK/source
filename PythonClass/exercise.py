import urllib.request           # 웹브라우저에서 html 문서 얻을 때 통신하기 위한 모듈
from  bs4 import BeautifulSoup  # html 문서 검색 모듈

def fetch_list_url():
        list_url = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq=0"
        url = urllib.request.Request(list_url)                   # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8") # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
                                                                 # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
        # 문자를 담는 set (p256 뇌자극)
        # 1. US7ASCII : 영어
        # 2. 유니코드(UTF8) : 한글, 중국어
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        soup2 = soup.find_all('p')                # p 태그의 첫번째 요소만 불러오기

        for link in soup2 :
            print(link.find('a')['href'])         # a 태그 안의 href 부분만 불러오기
        return

fetch_list_url()

