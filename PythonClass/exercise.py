import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def pdf_list(dirname):
    filenames = os.listdir(dirname)                     # 지정된 폴더 내 파일이름들 불러오기
    music_dict = {}
    for filename in filenames:
        full_filename = os.path.join(dirname, filename) # full_filename = 경로+파일이름
        ext = os.path.splitext(full_filename)[-1]       # ext에 확장자 넣기
        file = os.path.splitext(filename)[0]            # file에 확장자를 제외한 파일이름만 넣기
        if ext == '.mp3':                               # 확장자가 mp3 인 파일만 music_dict 딕셔너리에 넣기
            music_dict[file] = full_filename            # 파일이름(key), 경로+파일이름(value)
    return music_dict                                   # music_dict 딕셔너리 리턴


print(convert_pdf_to_txt('d://italianjob.pdf'))

ital = open('d://italianjob.txt', 'w', encoding='UTF-8', newline='')
ital.write(convert_pdf_to_txt('d://italianjob.pdf'))
ital.close()