import os
import shutil
import requests
from urllib.request import urlretrieve
import urllib.request
from bs4 import BeautifulSoup

total_count = 0
root_url = 'http://www.getchu.com'
y_m_url = 'http://www.getchu.com/all/month_title.html'

years = [str(y) for y in list(range(2015, 2020))]
months = [str(m).zfill(2) for m in list(range(1, 13))]

root_dir = './images'
#shutil.rmtree(root_dir, ignore_errors=True)
#os.mkdir(root_dir)

payload = {
	'gage': 'all',
	'gc': 'gc' # important
}

headers = {
	'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
	'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0"
}

types = ['pc_soft', 'dvd_game']

max_retries = 5

getchu_str = 'getchu_adalt_flag=getchu.com'
item_str = 'ITEM_HISTORY='
game_list = ''
cookies = getchu_str + '; ' + item_str
# give a header to the retriever (thanks, https://stackoverflow.com/a/46511429)
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'), ('Cookies', cookies)]
urllib.request.install_opener(opener)


first_cookie = True
for y in years:
	ct = 1
	out_dir = os.path.join(root_dir, y)
	try:
		os.mkdir(out_dir)
	except:
		pass
	for m in months:
		print("Scraping images in year {}, month {}".format(y, m))
		for t in types:
			success = False
			retries = 0
			while not success:
				try:
					by_year_month_res = requests.get(y_m_url, params = {**payload, 'year': y, 'month': m, 'genre': t}, headers=headers)
					year_month_soup = BeautifulSoup(by_year_month_res.text, 'html.parser')
					game_elems = year_month_soup.find_all('td', class_ = 'dd')
					for game in game_elems:
						game_ref = game.find('a').attrs['href']
						game_url = root_url + game_ref
						success = False
						retries = 0
						while not success:
							try:
								game_page_res = requests.get(game_url, params = {'gc': 'gc'}, headers=headers)
								game_page_soup = BeautifulSoup(game_page_res.text, 'html.parser')
								game_id = ''.join(list(filter(lambda x: x.isdigit(), game_url)))
								if (first_cookie == True):
									game_list = str(game_id)
									# cookies = cookies + game_id
									first_cookie = False
								else:
									game_list = str(game_id) + '%7C' + game_list
									# cookies = cookies + '%7C' + game_id
								# reset header
								game_list = game_list[0:(len(game_list) if len(game_list) < 1024 else 1023)]
								cookies = getchu_str + '; ' + item_str + game_list
								opener = urllib.request.build_opener()
								opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'), ('Cookies', cookies), ('Referer', game_url)]
								urllib.request.install_opener(opener)
								img_tags = game_page_soup.find_all('img', attrs = { 'alt': lambda x : x and 'キャラ' in x})
								character_tags = [root_url + tag.attrs['src'][1:] for tag in img_tags]
								for character in character_tags:
									urlretrieve(character, os.path.join(out_dir, '{}_{}.jpg'.format(y, ct)))
									ct += 1
								total_count += len(character_tags)
								print("Total images: {}".format(total_count))
								success = True
							except Exception as e:
								print("Fetch url {} fail!".format(game_url))
								print(e)
								print(game_page_res)
								retries += 1
								if retries == max_retries:
									success = True
					success = True
				except Exception:
					print("Fetch url {} fail!".format(y_m_url))
					retries += 1
					if retries == max_retries:
						success = True
		


soup = BeautifulSoup(requests.get('http://www.getchu.com/soft.phtml?id=727363&gc=gc').text, 'html.parser')
print(soup.find_all('img', attrs = { 'alt': lambda x : x and 'キャラ' in x}))