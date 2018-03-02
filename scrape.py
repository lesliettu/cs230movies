import urllib
from bs4 import BeautifulSoup
import json
import csv
import time

def pretty_print(dict):
	print(json.dumps(dict, sort_keys=True, indent=4, separators=(',', ': ')))

def array_to_text(arr):
	s = ""
	for item in arr:
		s += item + " "
	return s.strip()

def get_example(url):
	soup = BeautifulSoup(urllib.request.urlopen(url), "html.parser")

	example = { 'url': url }

	title_tag = soup.find('meta', attrs={'name': 'movieTitle'})
	example['title'] = title_tag.get('content') if title_tag else None

	year_tag = soup.find('span', attrs={'class': 'h3 year'})
	example['year'] = year_tag.text.strip() if year_tag else None

	poster_tag = soup.find('img', attrs={'class': 'posterImage'})
	example['poster'] = poster_tag.get('src') if poster_tag else None

	description_tag = soup.find('meta', attrs={'property': 'og:description'})
	example['description'] = description_tag.get('content') if description_tag else None

	audience_score_tag = (soup.find('div', attrs={'class': 'audience-score meter'}) 
	    					  .find('span', attrs={'class': 'superPageFontColor'}))
	example['audience_score'] = audience_score_tag.text.strip() if audience_score_tag else None

	critic_consensus_tag = soup.find('p', attrs={'class': 'critic_consensus superPageFontColor'})
	example['critics_consensus'] = ([s for s in critic_consensus_tag.stripped_strings][-1]
									if critic_consensus_tag else None)

	script_tag = soup.find('script', attrs={'id': 'jsonLdSchema'})
	if script_tag:
		meta_dict = json.loads(script_tag.text)
		if 'aggregateRating' in meta_dict:
			example['reviews_counted'] = meta_dict['aggregateRating']['reviewCount']
			example['percent_fresh'] = meta_dict['aggregateRating']['ratingValue']
		example['rating'] = meta_dict['contentRating']
		example['genre'] = array_to_text(meta_dict['genre'])

	return example

examples = []
failed_urls = []
count = 1
BATCH_SIZE = 500
batch_num = 0
REPORT_INTERVAL = 20
START = 2001
with open('rottentomatoes_urls.csv', 'r') as url_file:
	reader = csv.DictReader(url_file)
	for row in reader:
		count += 1
		if count < START: 
			continue 

		time.sleep(1) # sleep for one second so the internet gods don't hate me for scraping too fast

		try:
			examples.append(get_example(row['url']))
		except:
			print(':( Failed on',row['url'])
			failed_urls.append(row['url'])

		if count % REPORT_INTERVAL == 0:
			print('Scraped', count, 'examples')
		if count % BATCH_SIZE == 0:
			with open('examples' + str(count-BATCH_SIZE+1) + '-' + str(count)+'.json', 'w') as outfile:
				json.dump(examples, outfile)
			examples = []
			batch_num += 1

print(failed_urls)


# https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error
