import Classify
import bs4
from urllib.request import urlopen
import lxml
import re
import string
import time
import pandas as pd


links = []
reuters = 'https://www.reuters.com/markets'

while (True):
    page = urlopen(reuters)
    soup = bs4.BeautifulSoup(page, 'html.parser')
    link_list = []

    for link in soup.findAll('a', attrs={'href': re.compile("/article/")}):
        if bool(re.search('reuters', link.get('href'))) == True:
            link_clean = re.sub('https://www.reuters.com', '', link.get('href'))
            link_list.append(link_clean)
        else:
            link_clean = re.sub('\?il=0$', '', link.get('href'))
            link_list.append(link_clean)

    page_link_list = list(set(link_list))
    
    for i in set(page_link_list).difference(set(links)):
        x = Classify.ClassifyText(reuters, i, named_ent=True)
        print(x)
        time.sleep(5)

    for link in page_link_list:
        links.append(link)

    time.sleep(3)