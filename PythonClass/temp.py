from bs4 import BeautifulSoup

with open(r"c:\python\data\abc.html") as aaa:
    soup = BeautifulSoup(aaa,"lxml")

print(soup.find_all('a'))

