import konlpy
import numpy as np
from konlpy.tag import Kkma
from konlpy.utils import pprint

from konlpy.tag import Kkma, Mecab, Hannanum, Twitter
from konlpy.corpus import kolaw, kobill
from konlpy.utils import pprint
from nltk import collocations


kkma = Kkma()
hanna = Hannanum()
twitter = Twitter()
# pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
#
# pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
#
# pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))

# pprint(hanna.nouns(u'교육부 고위공무원인 갑이 신문기자들 등과의 식사와 음주 자리에서 민중은 개, 돼지다. 신분제 공고화해야 한다’는 등 취지의 발언을 하고 기사화에 따른 문제 발생과 파장이 예견되는 상황임에도 안이하게 대처하다가 언론에 보도됨으로써 공무원 품위유지의무를 위반하였다는 이유로, 교육부장관이 갑을 파면한 사안에서, '
#                 u'갑이 위와 같은 취지의 발언을 하고 안이하게 대처함으로써 공무원으로서의 품위유지의무를 위반하였으나 파면처분은 '
#                 u'갑의 비위행위의 정도에 비하여 지나치게 과중하여 비례의 원칙을 위반한 것으로 재량권의 한계를 벗어나 위법하다고 한 사례'))
# pprint(mecab.nouns(u'교육부 고위공무원인 갑이 신문기자들 등과의 식사와 음주 자리에서 ‘민중은 개, 돼지다. ' ))
# measures = collocations.BigramAssocMeasures()
# doc = kolaw.open('constitution.txt').read()
# print('\nCollocations among tagged words:')
# tagged_words = Kkma().pos(doc)
# finder = collocations.BigramCollocationFinder.from_words(tagged_words)
# pprint(finder.nbest(measures.pmi, 10)) # top 5 n-grams with highest PMI
#
# print('\nCollocations among words:')
# words = [w for w, t in tagged_words]
# ignored_words = [u'안녕']
# finder = collocations.BigramCollocationFinder.from_words(words)
# finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words)
# finder.apply_freq_filter(3) # only bigrams that appear 3+ times
# pprint(finder.nbest(measures.pmi, 10))
#
# print('\nCollocations among tags:')
# tags = [t for w, t in tagged_words]
# finder = collocations.BigramCollocationFinder.from_words(tags)
# pprint(finder.nbest(measures.pmi, 5))

# a = hanna.nouns(u'국립대학교 교수 갑이 강의 중 노무현은 전자개표기 사기극으로 당선된 가짜대통령이다. '
#                 u'자네들이 노무현 전자개표기 사기극 사건을 맡은 대법관이라면 어떻게 판결문을 쓸 것인지 리포트로 제출하라.”라고 발언하고, '
#                 u'인터넷 사이트에 유사한 내용의 게시물을 게재한 행위에 대하여 총장이 징계위원회의 의결에 따라 갑에게 파면처분을 한 사안에서, '
#                 u'파면처분이 객관적으로 부당하거나 사회통념상 현저하게 타당성을 잃을 정도로 갑에게 지나치게 가혹하여 재량권의 범위를 일탈·남용한 것이 아니라고 한 사례')
#
# b = np.array(a)
#
# c= twitter.nouns(u'국립대학교 교수 갑이 강의 중 노무현은 전자개표기 사기극으로 당선된 가짜대통령이다. '
#                 u'자네들이 노무현 전자개표기 사기극 사건을 맡은 대법관이라면 어떻게 판결문을 쓸 것인지 리포트로 제출하라.”라고 발언하고, '
#                 u'인터넷 사이트에 유사한 내용의 게시물을 게재한 행위에 대하여 총장이 징계위원회의 의결에 따라 갑에게 파면처분을 한 사안에서, '
#                 u'파면처분이 객관적으로 부당하거나 사회통념상 현저하게 타당성을 잃을 정도로 갑에게 지나치게 가혹하여 재량권의 범위를 일탈·남용한 것이 아니라고 한 사례')
# print(b)
# print(np.array(c))
#
# d = hanna.nouns(u'성폭력범죄의처벌등에관한특례법위반(13세미만미성년자강제추행'
#                 u'[1] 무죄추정의 원칙의 의의 / 형사재판에서 유죄를 인정하기 위한 증거의 증명력 정도 / 낮 시간대 다수의 사람들이 통행하는 공개된 장소와 같이 통상적으로 어린 피해자에 대한 추행 행위가 이루어질 것으로 '
#                 u'예상하기 곤란한 상황에서 강제 추행이 있었는지를 판단하는 데 피해자의 진술 또는 피해자와 밀접한 관계에 있는 자의 진술이 유일한 증거인 경우, '
#                 u'이를 근거로 피고인을 유죄로 판단하기 위한 진술의 신빙성 정도[2] ‘추행’의 의미 및 추행에 해당하는지 판단하는 기준')
# print(np.array(d))

# d = kobill.open('1809890.txt').read()
# print(d)


import re
import string

frequency = {}
document_text = open('C:\\python\\source\\Portpolio\\autocomplete\\data\\대한민국헌법.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z가-힣]{3,15}\b', text_string)

for word in match_pattern:
    count = frequency.get(word, 0)
    frequency[word] = count + 1

frequency_list = frequency.keys()


for words in frequency_list:
    print(words, frequency[words])


#########################################

from collections import Counter
import urllib
import random
import webbrowser

from konlpy.tag import Hannanum
from lxml import html
import pytagcloud # requires Korean font support
import sys

if sys.version_info[0] >= 3:
    urlopen = urllib.request.urlopen
else:
    urlopen = urllib.urlopen


r = lambda: random.randint(0,255)
color = lambda: (r(), r(), r())

def get_bill_text(billnum):
    url = 'http://pokr.kr/bill/%s/text' % billnum
    response = urlopen(url).read().decode('utf-8')
    page = html.fromstring(response)
    text = page.xpath(".//div[@id='bill-sections']/pre/text()")[0]
    return text

def get_tags(text, ntags=50, multiplier=10):
    h = Hannanum()
    nouns = h.nouns(text)
    count = Counter(nouns)
    return [{ 'color': color(), 'tag': n, 'size': c*multiplier }\
                for n, c in count.most_common(ntags)]

def draw_cloud(tags, filename, fontname='Noto Sans CJK', size=(800, 600)):
    pytagcloud.create_tag_image(tags, filename, fontname=fontname, size=size)
    webbrowser.open(filename)


bill_num = '1904882'
text = get_bill_text(bill_num)
tags = get_tags(text)
print(tags)
draw_cloud(tags, 'wordcloud.png')