# Import packages
import pandas as pd
import urllib
import re
import numpy as np

# Collect HTML of CIA World Factbook - World Languages webpage
webpage = urllib.request.urlopen('https://www.cia.gov/the-world-factbook/field/languages/')
webcontent = webpage.read().decode('utf-8')
# Split to main part of webpage that contains countries and languages
webcontent = webcontent.split("starting with the most-spoken language.")[1]
webcontent = webcontent.split("<footer>")[0]

# Parse HTML for country names and create a dataframe of countries
countries = re.findall(">([\w\s\-,\)\(]*)</a>", webcontent)
countries.remove("World")
countriesdf = pd.DataFrame(countries, columns = ["country"])

# Split HTML after each country
    # Each list in 'afterCountries' contains HTML of the languages of a country
afterCountries = re.split(">[\w\s\-,\)\(]*</a></h3><p>", webcontent)[1:]
afterCountries.pop(235)

# Parse each list in 'afterCountries' for language names
languages = []
for x in afterCountries:
    x = x.split("</p>")[0]
    x = x.split("note:")[0]
    x = re.sub('\([^\(]+\)|\sand\s|,|\sor\s|\d*\.*\d*%|<.*>|;|\d+|\/|\s[a-z]+|&nbsp|other|:|\(|&lt|languages|dialects|includes|used|official|judicial|national|indigenous|unspecified|unknown|local', '@', x)
    x = [y.strip() for y in x.split('@') if len(y)>1]
    languages.append(x)

# Retrieve language percentages (if provided)
percentages = []
i = 0
for list in languages:
    c = afterCountries[i]
    perc_list = []
    lang = 0
    while (lang < len(list)):
        c = afterCountries[i]
        y = str(list[lang])
        c = c.split(y)[1]
        if (lang < len(list)-1):
            z = str(list[lang+1])
            c = c.split(z)[0]
        result = re.search('\d+\.*\d*\%', c)
        if (result == None):
            perc_list.append(result)
        else:
            perc_list.append(result.group())
        lang += 1
    percentages.extend(perc_list)
    i += 1
   
# Create a dataframe of percentages
percentages = pd.DataFrame(percentages)

# Createa a dataframe with languages as lists
languagesdf = pd.DataFrame(np.zeros((238,1)))
languagesdf = languagesdf.astype('object')
a = 0
for x in languages:
    languagesdf[0][a] = x
    a = a + 1
    
# Join countries and languages dataframes (with languages as lists)
languageListsByCountry = countriesdf.join(languagesdf, rsuffix='_diag', how='right')
languageListsByCountry.columns = ['country', 'language']

# Createa a dataframe 'languagesByCountry' with each language in a separate cell
lst_col = 'language'
languagesByCountry = pd.DataFrame({col:np.repeat(languageListsByCountry[col].values, 
                                    languageListsByCountry[lst_col].str.len())
                                    for col in languageListsByCountry.columns.drop(lst_col)}
                                    ).assign(**{lst_col:np.concatenate(languageListsByCountry[lst_col].values)}
                                    )[languageListsByCountry.columns]\

# Add percentages column
languagesByCountry['Percentage'] = percentages

# Read in Arizona state refugee data
az_countries = pd.read_csv("azRefugeeNumbers.csv")
del az_countries['language']
az_countries = az_countries[["country", "total_since_1975", "total_since_2000", "total_since_2015"]]

# Join Arizona state refugee data with 'languagesByCountry' dataframe
az_and_languages = languagesByCountry.join(az_countries.set_index('country'), how='inner', on='country')
az_and_languages.reset_index(drop=True, inplace=True)
# Manually identify errors to drop from dataframe
az_and_languages = az_and_languages.drop([69, 70, 71, 72, 73, 106, 198, 214, 257, 258, 259, 260, 309, 310, 328])

# Calculate countries' weighted totals
weighted_countries = pd.DataFrame(columns=['country', 'weighted'])
country_list = []
weighted_country_list = []
i = 0
while i < 96:
    country_list.append(az_countries.iloc[i]['country'])
    weighted_country_list.append(int(az_countries.iloc[i]['total_since_2000']) / az_countries['total_since_2000'].sum())
    i += 1
weighted_countries['country'] = country_list
weighted_countries['weighted'] = weighted_country_list

# Create a ranked country list (descending)
countries_ranked = weighted_countries.sort_values(by='weighted', ignore_index=True)
countries_ranked.index = countries_ranked.index + 1
countries_ranked = countries_ranked.sort_index(ascending=False)

# 'Sum from' function
    # Sums numbers in a given range
    # Used to calculate languages' weighted totals
def sumfrom(a, b):
    result = 0
    while b >= a:
        result = result + b
        b -= 1
    return result

# Calculate languages' rank within country
language_ranked_in_country = []
total_lang_in_country = []
weighted_lang_per_c_list = []
weighted_lang_total_list = []
country_a = az_and_languages.iloc[0]['country']
country_b = az_and_languages.iloc[0]['country']
count = 1
total_lang = 0
i = 1
while i < len(az_and_languages):
    country_b = az_and_languages.iloc[i]['country']
    if country_a == country_b:
        count += 1
        if i == len(az_and_languages)-1:
            while count > 0:
                language_ranked_in_country.append(count)
                total_lang_in_country.append(total_lang)
                count -= 1
    else:
        total_lang = count
        while count > 0:
            language_ranked_in_country.append(count)
            total_lang_in_country.append(total_lang)
            count -= 1
        count = 1
        country_a = az_and_languages.iloc[i]['country']
    i += 1
    
# Add columns to 'az_and_languages' dataframe:
    # Language ranking within country
    # Total number of languages in country
az_and_languages['ranked_in_country'] = language_ranked_in_country
az_and_languages['total_lang_per_country'] = total_lang_in_country

# Calculate each language's weight within each country:
    # If a percentage was parsed for the language, the percentage is used
    # If not, an artificial percentage is created based on the languages' rank within country
i = 0
while i < len(az_and_languages):
    perc = az_and_languages['Percentage'].iloc[i]
    if (perc == None):
        weighted_lang_per_c_list.append((az_and_languages.iloc[i]['ranked_in_country']) / sumfrom(1, 
                                            az_and_languages.iloc[i]['total_lang_per_country']))
    else:
        perc_dec = float(perc.strip('%'))/100
        weighted_lang_per_c_list.append(perc_dec)
    i += 1
    
# Add a column to 'az_and_languages' for each language's weight within each country
az_and_languages['weighted_lang_per_country'] = weighted_lang_per_c_list

# Calculate each language's overall weighted total, using its weight per country
    # and the country's weighted total of refugees
i = 0
while i < len(az_and_languages):
    weighted_lang_total_list.append((az_and_languages.iloc[i]['weighted_lang_per_country']) 
                            * ((az_and_languages.iloc[i]['total_since_2000']) / az_countries['total_since_2000'].sum()))
    i += 1
    
# Add a column to 'az_and_languages' for each language's weighted total
az_and_languages['weighted_lang_total'] = weighted_lang_total_list

# Create language rankings by summing all weighted totals for each occurrence of a language
languages_ranked = pd.DataFrame(columns=['languages', 'proportion_of_total'])
lang = []
languages_proportion = []
k = 0
total_prop = 0
while k < len(az_and_languages):
    lang_a = az_and_languages.iloc[k]['language']
    if lang_a not in lang:
        i = k
        while i < len(az_and_languages):
            lang_b = az_and_languages.iloc[i]['language']
            if lang_a == lang_b:
                total_prop = total_prop + az_and_languages.iloc[i]['weighted_lang_total']
            i += 1
        lang.append(lang_a)
        languages_proportion.append(total_prop)
        total_prop = 0
    k += 1
    
# Create a dataframe of language rankings
languages_ranked['languages'] = lang
languages_ranked['proportion_of_total'] = languages_proportion
languages_ranked = languages_ranked.sort_values(by='proportion_of_total', ignore_index=True)
languages_ranked.index = languages_ranked.index + 1
languages_ranked = languages_ranked.sort_values(by='proportion_of_total', ascending=False)

# Reset index to start at 1 instead of 0
languages_ranked = languages_ranked.sort_values(by='proportion_of_total')
languages_ranked.reset_index(drop=True, inplace=True)
languages_ranked.index = languages_ranked.index+1
languages_ranked = languages_ranked.sort_index(axis=0, ascending=False)
languages_ranked['languages'] = languages_ranked['languages'].replace(['- Tagalog'], 'Tagalog')

# Address weighted total ties between languages and assign rankings
ascend = languages_ranked.sort_index(ascending=False)
rank = []
track = 1
r = 0
prev_val = -5
for i in ascend.itertuples():
    val = i.proportion_of_total
    if (val == prev_val):
        rank.append(r)
    else:
        r += track
        rank.append(r)
        track = 0
    prev_val = i.proportion_of_total
    track += 1
ascend['rank'] = rank

# Sort languages by ranking, in ascengind order (Highest ranked refugee language has ranking=1)
ascend = ascend.sort_values(by='rank', ascending=True)
ascend

# Output language rankings to csv
ascend.to_csv('languages_ranked.txt', sep='\t')

# Import patient data (Patient ID, age, and refugee status)
lang_stat = pd.read_csv('langs_and_stat.csv')

# Read in language ranking file without index column
languages_ranked = pd.read_csv('languages_ranked.txt', sep='\t', index_col=False)
del languages_ranked['Unnamed: 0']

# Manually rename languages so that patient data and language rankings list use same terminology
lang_stat['languages'] = lang_stat['languages'].replace(['Swahili; Kiswahili'], 'Swahili')
lang_stat['languages'] = lang_stat['languages'].replace(['Spanish/English'], 'Spanish')
lang_stat['languages'] = lang_stat['languages'].replace(['Farsi; Persian'], 'Dari')
lang_stat['languages'] = lang_stat['languages'].replace(['Chinese (Mandarin)'], 'Mandarin')

# Merge patient data and language rankings dataframes
pred_stat = lang_stat.merge(languages_ranked, how='left', on='languages')
del pred_stat['proportion_of_total']

# "Other" ranking: fill missing rankings with ranking=50
pred_stat.fillna(50, inplace=True)

# Remove 'languages' and 'Refugee status' columns
    # Result is a dataframe with Patient ID and language ranking only
del pred_stat['languages']
del pred_stat['Refugee status']

# Rename rank column and output dataframe to csv as 'language_rank'
pred_stat = pred_stat.rename(columns={'rank':'language_ranking'})
pred_stat.to_csv('language_rank.csv', sep=',', index=False)