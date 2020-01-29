import time
import pandas as pd
from selenium import webdriver

# Scrapping images and their caption from unsplash website
# saving these images and captions into a csv file

WEBSITE = 'http://unsplash.com/s/photos/landscape'
columns = ['description', 'url']
imageset = pd.DataFrame(columns = columns)

# Define Chrome options to open the window in maximized mode
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

# Initialize the Chrome webdriver and open the URL
driver = webdriver.Chrome(options=options)

# get web page
driver.get(WEBSITE)
#Define a pause time in between scrolls
pause_time = 5
slow=1000
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

# Create an emply list to hold all the urls for the images
hrefs = []
title = []
dict = {}
image_to_scrap = 2000
# We only want to scrap images of landscapes without peaople
tagToAvoid = ['man', 'person', 'people', 'road', 'woman']

while (len(dict) < image_to_scrap):
    # Scroll par etapes
    driver.execute_script("window.scrollTo(0, "+str(slow)+");")
    slow = slow + 1000
    #on recupere la hauteur a scroller
    last_height = driver.execute_script("return document.body.scrollHeight")
    # wait to load page
    time.sleep(pause_time)
    # Calculate new scroll height and compare with last scroll height

    if slow >= last_height: # which means end of page
        break
    # Extract all anchor tags
    link_tags = driver.find_elements_by_class_name("_2zEKz");
    # Extract the urls and titles of only the images from each of the tag WebElements
    for tag in link_tags:
        #to avoid duplicate, use of a dictionnary
        if((tag.get_attribute('src') not in dict) and tag.get_attribute('alt') and len(tag.get_attribute('alt')) > 10):
            if((tag.get_attribute('alt').find('man') == -1)
                    and (tag.get_attribute('alt').find('men') == -1)
                    and (tag.get_attribute('alt').find('person') == -1)
                    and (tag.get_attribute('alt').find('people') == -1)
                    and (tag.get_attribute('alt').find('road') == -1)) :
                dict[tag.get_attribute('src')] = tag.get_attribute('alt')
                hrefs.append(tag.get_attribute('src'))
                title.append(tag.get_attribute('alt').replace(',',''))
    print('height scroll :',last_height, '\tslow :',slow, '\tlen dict:',len(dict))

print(len(hrefs), len(title))
imageset.loc[:,'description'] = title
imageset.loc[:,'url'] = hrefs
imageset.to_csv(r'Data\landscapeSet_v3.csv')
# Select all duplicate rows based on multiple column names in list
duplicateRowsDF = imageset[imageset.duplicated(['description', 'url'])]

print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')