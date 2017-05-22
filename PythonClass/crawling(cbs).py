import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
from  bs4 import BeautifulSoup  # html 문서 검색 모듈
import os
import re


def get_save_path():
    save_path = input("Enter the file name and file location :")
    save_path = save_path.replace("\\", "/")
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])
    return save_path


def fetch_list_url(page):
    params = []
    for j in range(1, page + 1):
        list_url = "http://www.cbs.co.kr/radio/pgm/board.asp?page={}&pn=list&" \
                   "skey=&sval=&bgrp=2&bcd=00350012&" \
                   "pcd=board&pgm=111&mcd=BOARD2".format(j)
        url = urllib.request.Request(list_url)
        res = urllib.request.urlopen(url).read()
        # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
        soup = BeautifulSoup(res, "html.parser")

        for i in range(15):
            soup2 = soup.find_all('a', class_="bd_link")[i]['href']
            soup3 = re.search("[0-9,',']{11}", soup2).group().split(',')
            params.append(soup3)
    # print(params)
    return params


# fetch_list_url(3)

def fetch_list_url2(page):
    params2 = fetch_list_url(page)
    # print(params2)

    f = open(get_save_path(), 'w', encoding="utf-8")

    for list_url in params2:
        url1 = "http://www.cbs.co.kr/radio/pgm/board.asp?pn=read&skey=&" \
               "sval=&anum=" + list_url[1] + "&vnum=" + list_url[0] + "&bgrp=2&page=1" \
                                                                      "&bcd=00350012&pcd=board&pgm=111&mcd=BOARD2"
        url2 = urllib.request.Request(url1)
        res = urllib.request.urlopen(url2).read()
        soup = BeautifulSoup(res, "html.parser")
        content = soup.find('td', id="BoardContents").get_text(strip=True, separator='\n')
        title = soup.find_all('td', class_='bd_menu_content')[0].get_text(strip=True, separator='\n')
        date = soup.find_all('td', class_='bd_menu_content')[3].get_text(strip=True, separator='\n')
        f.write(title)
        f.write('\n')
        f.write(date)
        f.write('\n')
        f.write(content)
        f.write('\n\n')
    f.close()


fetch_list_url2(3)