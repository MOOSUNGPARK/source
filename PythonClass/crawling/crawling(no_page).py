import urllib.request           # 웹브라우저에서 html문서를 얻어오기 위해 통신하기 위한 모듈
from bs4 import BeautifulSoup   # html문서 검색 모듈
import os
import re


def get_save_path():
    save_path = input('Enter the file name and file location : ')
    save_path = save_path.replace('\\', '/')

    # 폴더가 없으면 폴더를 만드는 작업
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])

    return save_path


def fetch_list_url(page):
    params = []

    for cnt in range(1,page+1):
        list_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp".format(cnt)
                    ###여기에 스크롤링할 웹페이지의 url 을 붙여넣습니다.
        request_header = urllib.parse.urlencode({"page":cnt})           # page=1
        request_header = request_header.encode("utf-8")                 # b'page=1' 이런 형식으로 바꿔줘야 함

        url = urllib.request.Request(list_url, request_header)          # url 요청에 따른 http 통신 헤더값을 얻어낸다
                                                                        # request_header 를 써줘야 해당 페이지 url 불러옴
                                                                        # <urllib.request.Request object at 0x0000000001DCA0F0> 이런 정보 불러옴
        res = urllib.request.urlopen(url).read().decode("utf-8")        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
                                                                        # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다. (유니코드 안쓰면 글씨 다 깨짐)
        # < a href = "JavaScript:onView ('20170504000603')" title = "주택가 무속인 미취학 아동 보호 " > 주택가 무속인 미취학 아동보호 < / a >
        # 여기서 onView함수 안의 20170504000603 을 불러와야 함

        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정

        for link in soup.find_all('li', class_='pclist_list_tit2'):
            try:

                params.append(re.sub('[^0-9]','',link.a["href"]))   # link.a["href"] = link.find('a')[href]
                        # '20170504000603'
                # params.append(re.search('[0-9]{14}', link.a["href"]).group()) # 이렇게 해도 됨.
                        # '20170504000603'
            except:  #<dt></dt>처럼 비어있는 곳이 있기 때문에 try except를 사용해야 한다.
                continue
    # print(params)
    return params

def fetch_list_url2(page):
    params2 = fetch_list_url(page)
    f = open(get_save_path(),'w',encoding='utf-8')
    format_dic = {0: '============ 제목 ============',
                  1: '============ 날짜 ============',
                  2: '============ 민원 ============',
                  3: '============ 답변 ============',
                  4: '='*50+'\n'}

    for param in params2:
        list_url = 'http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
        request_header = urllib.parse.urlencode({'RCEPT_NO' : param })
        request_header = request_header.encode('utf-8')

        url = urllib.request.Request(list_url, request_header)  # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")  # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
        soup = BeautifulSoup(res, "html.parser")  # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정

        soup2 = soup.find('div', class_='form_table')
        soup3 = soup2.find_all('td')

        for idx, script in enumerate(soup3):
            f.write(format_dic[idx] + '\n' + script.get_text(strip=True, separator='\n')+'\n')
        f.write(format_dic[4]+'\n')
    f.close()

fetch_list_url2(50)

