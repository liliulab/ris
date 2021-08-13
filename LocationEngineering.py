# Import packages
from arcgis.gis import GIS
from arcgis.geocoding import get_geocoders, batch_geocode, geocode
import pandas as pd

# Create map of Phoenix area
gis = GIS()
geocoder = get_geocoders(gis)[0]
map = gis.map("Phoenix, AZ", 10)
map

# Read in file containing patients' ID, address, and refugee status
df = pd.read_csv('add_and_stat.csv', engine='python')


# Mapping Addresses ..........................................................

# Import packages
from geopy.geocoders import Nominatim
from geopy import distance
geolocator = Nominatim(user_agent='geoapiExercises')

#                    Map All Patients ........................................

# Create list of addresses
addresses = []
i = 0
while i < len(df):
    addresses.append(df.iloc[i]['ADD_LINE_1']+', '+df.iloc[i]['CITY']+', AZ '+df.iloc[i]['Zip_Code'])
    i += 1
    
# Plot addresses on the map
for a in addresses:
    ad = geocode(a)[0]
    popup = {
        'title' : 'a',
        'content' : ad['address']
    }
    map.draw(ad['location'], popup)

# Find centroid of dataset, to impute 'None' values with
latitudes_c = []
longitudes_c = []
locations_c = []
for a in addresses:
    place_c = geolocator.geocode(a)
    if (place_c != None):
        lat_c, lon_c = (place_c.latitude), (place_c.longitude)
        location_c = (lat_c, lon_c)
        latitudes_c.append(lat_c)
        longitudes_c.append(lon_c)
        locations_c.append(location_c)
cent_lat = sum(latitudes_c) / len(latitudes_c)
cent_lon = sum(longitudes_c) / len(longitudes_c)
cent = (cent_lat, cent_lon)

# Create 'locations' list of coordinates
num_nones = 0
latitudes = []
longitudes = []
locations = []
for a in addresses:
    place = geolocator.geocode(a)
    if (place != None):
        lat, lon = (place.latitude), (place.longitude)
    else:
        lat, lon = cent_lat, cent_lon
        num_nones += 1
    location = (lat, lon)
    latitudes.append(lat)
    longitudes.append(lon)
    locations.append(location)

#                       Map Non-Refugees .....................................

# Create list of non-refugee addresses
non_ref_add = df.loc[(df['Refugee status'].str.lower() == 'no'), ['ADD_LINE_1', 'CITY', 'Zip_Code']]
non_ref_addresses = []
i = 0
while i < len(non_ref_add):
    non_ref_addresses.append(non_ref_add.iloc[i]['ADD_LINE_1']+', '+non_ref_add.iloc[i]['CITY']+', AZ '+non_ref_add.iloc[i]['Zip_Code'])
    i += 1
    
# Plot non-refugee addresses on the map
for a in non_ref_addresses:
    ad = geocode(a)[0]
    popup = {
        'title' : 'a',
        'content' : ad['address']
    }
    map.draw(ad['location'], popup)
    
# Find centroid of non-refugees dataset, to impute 'None' values with
nr_latitudes_c = []
nr_longitudes_c = []
nr_locations_c = []
for a in non_ref_addresses:
    nr_place_c = geolocator.geocode(a)
    if (nr_place_c != None):
        nr_lat_c, nr_lon_c = (nr_place_c.latitude), (nr_place_c.longitude)
        nr_location_c = (nr_lat_c, nr_lon_c)
        nr_latitudes_c.append(nr_lat_c)
        nr_longitudes_c.append(nr_lon_c)
        nr_locations_c.append(nr_location_c)
nr_cent_lat = sum(nr_latitudes_c) / len(nr_latitudes_c)
nr_cent_lon = sum(nr_longitudes_c) / len(nr_longitudes_c)
nr_cent = (nr_cent_lat, nr_cent_lon)

# Create 'nr_locations' list of non-refugee coordinates
nr_latitudes = []
nr_longitudes = []
nr_locations = []
for a in non_ref_addresses:
    nr_place = geolocator.geocode(a)
    if (nr_place != None):
        nr_lat, nr_lon = (nr_place.latitude), (nr_place.longitude)
    else:
        nr_lat, nr_lon = nr_cent_lat, nr_cent_lon
    nr_location = (nr_lat, nr_lon)
    nr_latitudes.append(nr_lat)
    nr_longitudes.append(nr_lon)
    nr_locations.append(nr_location)
    
#                         Map Refugees .......................................

# Create list of refugee addresses
ref_add = df.loc[(df['Refugee status'].str.lower() == 'yes'), ['ADD_LINE_1', 'CITY', "Zip_Code"]]
ref_addresses = []
i = 0
while i < len(ref_add):
    ref_addresses.append(ref_add.iloc[i]['ADD_LINE_1']+', '+ref_add.iloc[i]['CITY']+', AZ '+ref_add.iloc[i]['Zip_Code'])
    i += 1
    
# Plot refugee addresses on the map
for a in ref_addresses:
    ad = geocode(a)[0]
    popup = {
        'title' : 'a',
        'content' : ad['address']
    }
    map.draw(ad['location'], popup)
    
# Find centroid of refugees dataset, to impute 'None' values with
r_latitudes_c = []
r_longitudes_c = []
r_locations_c = []
for a in ref_addresses:
    r_place_c = geolocator.geocode(a)
    if (r_place_c != None):
        r_lat_c, r_lon_c = (r_place_c.latitude), (r_place_c.longitude)
        r_location_c = (r_lat_c, r_lon_c)
        r_latitudes_c.append(r_lat_c)
        r_longitudes_c.append(r_lon_c)
        r_locations_c.append(r_location_c)
r_cent_lat = sum(r_latitudes_c) / len(r_latitudes_c)
r_cent_lon = sum(r_longitudes_c) / len(r_longitudes_c)
r_cent = (r_cent_lat, r_cent_lon)

# Create 'r_locations' list of refugee coordinates
r_latitudes = []
r_longitudes = []
r_locations = []
for a in ref_addresses:
    r_place = geolocator.geocode(a)
    if (r_place != None):
        r_lat, r_lon = (r_place.latitude), (r_place.longitude)
    else:
        r_lat, r_lon = r_cent_lat, r_cent_lon
    r_location = (r_lat, r_lon)
    r_latitudes.append(r_lat)
    r_longitudes.append(r_lon)
    r_locations.append(r_location)
    
    
# Plotting Dendrograms .......................................................

# Import packages
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

#                       Dendrogram of All Patients ...........................

# Utilize Ward's linkage method to cluster
linked = linkage(locations, 'ward')
labelList = range(1, len(locations)+1)

# Plot dendrogram
plt.figure(figsize=(20, 14))
dendrogram(linked, 
          orientation='top',
          labels=labelList,
          distance_sort='descending',
          show_leaf_counts=True)
plt.show()

#                       Dendrogram of Non-Refugees ...........................

# Utilize Ward's linkage method to cluster
nr_linked = linkage(nr_locations, 'ward')
nr_labelList = range(1, len(nr_locations)+1)

# Plot non-refugee dendrogram
plt.figure(figsize=(20, 14))
dendrogram(nr_linked, 
          orientation='top',
          labels=nr_labelList,
          distance_sort='descending',
          show_leaf_counts=True)
plt.show()

#                       Dendrogram of Refugees ...............................

# Utilize Ward's linkage method to cluster
r_linked = linkage(r_locations, 'ward')
r_labelList = range(1, len(r_locations)+1)

# Plot refugee dendrogram
plt.figure(figsize=(20, 14))
dendrogram(r_linked, 
          orientation='top',
          labels=r_labelList,
          distance_sort='descending',
          show_leaf_counts=True)
plt.show()


# Clustering .................................................................

# Import packages
from sklearn.cluster import AgglomerativeClustering

#                        Clustering All Patients .............................

# Plot address clusters
cluster = AgglomerativeClustering(n_clusters=20, affinity='euclidean', linkage='ward')
clust_pred = cluster.fit_predict(locations)
plt.xlim(-115, -110.5)
plt.ylim(32, 35.5)
plt.scatter(longitudes, latitudes, c=cluster.labels_, cmap='rainbow')

#                        Clustering Non_Refugees .............................

# Plot non-refugee address clusters
nr_cluster = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
nr_clust_pred = nr_cluster.fit_predict(nr_locations)
plt.xlim(-115, -110.5)
plt.ylim(32, 35.5)
plt.scatter(nr_longitudes, nr_latitudes, c=nr_cluster.labels_, cmap='rainbow')

#                        Clustering Refugees .................................

# Plot refugee address clusters
r_cluster = AgglomerativeClustering(n_clusters=11, affinity='euclidean', linkage='ward')
r_clust_pred = r_cluster.fit_predict(r_locations)
plt.xlim(-115, -110.5)
plt.ylim(32, 35.5)
plt.scatter(r_longitudes, r_latitudes, c=r_cluster.labels_, cmap='rainbow')


# Nearest Centroid ...........................................................

# Import packages
from sklearn.neighbors import NearestCentroid

#                     All Patients Nearest Centroid ..........................

# Identify each cluster's centroid
clf = NearestCentroid()
clf.fit(locations, clust_pred)
centroids = clf.centroids_

#                     Non-refugees Nearest Centroids .........................

# Identify each non-refugee cluster's centroid
nr_clf = NearestCentroid()
nr_clf.fit(nr_locations, nr_clust_pred)
nr_centroids = nr_clf.centroids_
print(nr_centroids)

#                     Refugee Nearest Centroids ..............................

# Identify each refugee cluster's centroid
r_clf = NearestCentroid()
r_clf.fit(r_locations, r_clust_pred)
r_centroids = r_clf.centroids_
print(r_centroids)


# Output Results .............................................................

#                     Output All Patients ....................................

# Create 'result' dataframe with patient ID and refugee status
result = pd.DataFrame()
result['Mother_MRN'] = df['Mother_MRN'].values
result['Refugee_status'] = df['Refugee status'].values

# Calculate distance from each point to each centroid and create new columns
    # Add to 'result'
    # Note: distance is in miles
i = 1
for c in centroids:
    j = 0
    result['dist_cent'+str(i)] = ''
    for l in locations:
        dist = distance.distance(l, c).mi
        result.loc[j, 'dist_cent'+str(i)] = dist
        j = j + 1
    i = i + 1
    
# Output 'result' to csv file
result.to_csv('centroid_distances.csv', sep=',', index=False)

#                      Output Non-Refugees ...................................

# Create 'nr_result' dataframe with patient ID and refugee status
nr_result = pd.DataFrame()
nr_result['Mother_MRN'] = df['Mother_MRN'].values
nr_result['Refugee_status'] = df['Refugee status'].values

# Calculate distance from each point to each centroid and create new columns
    # Add to 'nr_result'
i = 1
for c in nr_centroids:
    j = 0
    nr_result['nr_dist_cent'+str(i)] = ''
    for l in locations:
        nr_dist = distance.distance(l, c).mi
        nr_result.loc[j, 'nr_dist_cent'+str(i)] = nr_dist
        j = j + 1
    i = i + 1

# Output 'nr_result' to csv file
nr_result.to_csv('non_refugee_centroid_distances.csv', sep=',', index=False)

#                       Output Refugees ......................................

# Create 'r_result' dataframe with patient ID and refugee status
r_result = pd.DataFrame()
r_result['Mother_MRN'] = df['Mother_MRN'].values
r_result['Refugee_status'] = df['Refugee status'].values

# Calculate distance from each point to each centroid and create new columns
    # Add to 'r_result'
i = 1
for c in r_centroids:
    j = 0
    r_result['r_dist_cent'+str(i)] = ''
    for l in locations:
        r_dist = distance.distance(l, c).mi
        r_result.loc[j, 'r_dist_cent'+str(i)] = r_dist
        j = j + 1
    i = i + 1
    
# Output 'r_result' to csv file
r_result.to_csv('refugee_centroid_distances.csv', sep=',', index=False)