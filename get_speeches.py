import pandas as pd
import datetime as d
import os
import re
import requests
from bs4 import BeautifulSoup

# Create data directory if it doesn't exist
if os.path.isdir('data/') == False:
    os.system('mkdir data/')

# Function to scrape the speeches from the Fed website
def scrape_speeches():
    # Set the current year
    current_year = int(d.datetime.now().year)
    
    # Create list of years from 2006 to now
    years = list(range(2006, current_year))

    dfs = []
    for year in years:
        # Choose the proper webpage to access
    	print('Collecting data for year {}...'.format(year))
    	if year < 2011:
            page = 'https://www.federalreserve.gov/newsevents/speech/{}speech.htm'.format(year)
    	if year >= 2011:
            page = 'https://www.federalreserve.gov/newsevents/speech/{}-speeches.htm'.format(year)
    	# Access page
    	page_response = requests.get(page)
    	page_content = BeautifulSoup(page_response.content, 'html.parser')
    	# Collect all of the links
    	rows = page_content.find_all('div', attrs={'class':'row'})
    	speeches = rows[9]
    	# Get the links
    	links = speeches.find_all('a', attrs={'href':re.compile('\w+\d+\w.htm')}) # Use regular expression to filter out URLS
    	# Get all of the links to the articles
    	urls = []
    	for link in links:
            urls.append(link['href'])
    	# Begin the process of scraping and collecting the data
    	dates = []
    	speakers = []
    	locations = []
    	speech_text = []
    	base_url = 'https://www.federalreserve.gov'
    	print('Scraping speeches...'.format(year))
    	for url in urls:
            link = base_url+url
            # Access page
            page_response = requests.get(link)
            page_content = BeautifulSoup(page_response.content, 'html.parser')
            # Collect data and append to lists
            # dates
            date = page_content.find('p', attrs={'class':'article__time'}).text
            dates.append(date)
            # speakers
            speaker = page_content.find('p', attrs={'class':'speaker'}).text
            speakers.append(speaker)
            # locations
            location = page_content.find('p', attrs={'class':'location'}).text
            locations.append(location)
            # speech text
            block = page_content.find('div', attrs={'class':'col-xs-12 col-sm-8 col-md-8'})
            paragraphs = block.find_all('p')
            text_list = []
            for paragraph in paragraphs:
                text_list.append(paragraph.text)
            text = ' '.join(text_list)
            speech_text.append(text)
            # Save info as a dataframe
            df = pd.DataFrame(data={'Date':dates, 'Speaker':speakers, 'Location':locations, 'Speech':speech_text})
    	# Append data frame to the main loop
    	dfs.append(df)
    	print('{} is done!'.format(year))
    # Combine every year to form dataset
    data = pd.concat(dfs, ignore_index = True)
	
    # Save the dataset
    data.to_json('data/fed_speeches.json', orient='records')

# Execute function from command line
if __name__ == '__main__':
	scrape_speeches()
