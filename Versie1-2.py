#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install opendatasets
#!pip install matplotlib

# In[2]:


import numpy as np 
import pandas as pd 
import requests
import datetime as dt
from datetime import datetime 
import matplotlib.pyplot as plt 
import folium
import plotly.express as px
import seaborn as sns
import json
import geopandas
import ipywidgets as widgets


# In[3]:


od.download('https://www.kaggle.com/datasets/warcoder/earthquake-dataset')


# In[4]:


od.download('https://www.kaggle.com/datasets/aradzhabov/earthquakes-solar-system-objects')


# In[5]:


solar = pd.read_csv('SolarSystemAndEarthquakes.csv')


# In[6]:


solar.head()


# In[7]:


df = pd.read_csv('earthquake_data.csv')


# In[8]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[9]:


df.columns


# In[10]:


#cdi = the maximum reported intensity for the event range 
#mmi = the maximum estimated instrumental intensity for the event 
#alert = the alert level - 'green', 'yellow', 'orange' and 'red'
#tsunami = '1' for events in oceanic regions and '0' otherwise
#sig = a number describing how significant event is. larger numbers indicate a more significant event
#value is determined on a number of factors including mag, mmi
#net = the ID of a data contributer 
#nst = the total number of seismic stations used to determine earthquake location
#dmin = horizontal distance form the epicenter to the nearest station
#gap = the largest azimuthal gap between azimuthally adjacent station (in degrees)
#the smaller the number, the more reliable is the calculated horizontal
#position of the earthquake
#magtype = the method or algorithm used to calculate the preferred magnitude 
#depth = the depth where the earthquake begins to rupture

#"red" Estimated Fatalities 1,000+, Estimated Losses(USD) 1 billion+
#"orange" Estimated Fatalities 100 - 999, Estimated Losses(USD) 100 million - 1 billion
#"yellow" Estimated Fatalities 1-99, Estimated Losses(USD) 1 million - 100 million
#"green" Estimated Fatalities 0, Estimated Losses(USD) < 1 million


# In[11]:


df.isna().sum()
#veel missende values bij continent en country
#is op te lossen gezien we wel de lat en lon hebben 

#de alert column is waarschijnlijk welke waarschuwing de inwoners hebben gekregen? 


# In[12]:


df.shape
#782 rijen en 19 kolommen 


# In[13]:


df.dtypes


# In[14]:


df['Date'] = pd.to_datetime(df['date_time']).dt.date
df['Time'] = pd.to_datetime(df['date_time']).dt.time


# In[15]:


df['date_time'] =  pd.to_datetime(df['date_time'], format='%d%m%Y%I%M', errors = 'ignore')


# In[16]:


#raw_data['Mycol']=raw_data['Mycol'].astype('datetime64[ns]')
#convert to a float or int


# In[17]:


df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month


# In[18]:


df = df.drop(['date_time', 'Date', 'title'], axis=1)


# In[19]:


df = df.rename(columns = {'cdi':'reported intensity', 'mmi':'estimated intensity'})


# In[20]:


df['difference rep vs. est'] = df['reported intensity'] - df['estimated intensity']


# In[21]:


df['tsunami'] = df['tsunami'].replace({1: True, 0: False})


# In[22]:


df.head()


# In[23]:


corr_matrix = df[['magnitude', 'reported intensity', 'estimated intensity']]


# In[24]:


fig = px.box(df, x='continent', y='magnitude', title = 'Boxplot magnitude per continent with and without tsunami', color = 'tsunami')
fig.show()
#we zien hier een uitschieter van mag 8.8 in South America bij geen tsunami
#laagst gemeten mag is 6.5


#https://plotly.com/python/box-plots/


# In[25]:


df.describe()[['magnitude', 'reported intensity', 'estimated intensity']]


# In[26]:


#fig = px.bar(df, x='month', y='reported intensity')
#fig.show()


# In[27]:


#heeft de rupture depth van de aardbeving invloed op de accuratie van de magnitude voorspelling?


# In[28]:


fig = px.histogram(df, x='continent', color = 'net')

fig.show()
#the highest data contributer in Asia is the us.
#US - USGS National Earthquake Information Center, PDE (aka GS, NEIC)
#BRON: https://earthquake.usgs.gov/data/comcat/contributor/


# In[29]:


fig = px.scatter(df, x="year", y="magnitude", color="alert", color_discrete_sequence=["green", "yellow", "orange", "red"])
fig.show()
#hier zien we dat in 2010 een magnitude 'red' was afgegeven en in 2012 de hoogste magnitude 'yellow' was. 
#zegt deze data wel iets gezien hoogste magnitude eigenlijk 9.1 is? 

#SLIDER TOEVOEGEN? 


# In[30]:


#bij alert 'green' zijn de meeste tsunami's voorgekomen
df.groupby('alert')['tsunami'].sum()


# In[31]:



#fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="alert", size="magnitude",
#                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
#                  mapbox_style="carto-positron")
#fig.show()


# In[32]:


#fig = px.choropleth(df, locations="country", color="alert", hover_name="continent", animation_frame="year", range_color=[20,80])
#fig.show()


# In[33]:


df.head()


# In[34]:


df.groupby('year')['tsunami'].sum()
#hoe kan het dat er van 2001 tot en met 2012 geen tsunami's waren? foutieve data? 


# In[35]:


fig = px.bar(df, y="continent", x="gap", color="continent", orientation="h", hover_name="country",
             color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"],
             title="Gaps per continent"
            )

fig.show()
#in the continent Asia there is a lot of data about the gaps. Despite,in North America are the biggest gaps measured.
#biggest gaps: Mexico 239 degrees
#gap = the largest azimuthal gap between azimuthally adjacent station (in degrees)


# In[36]:


biggest_gap = df[df["gap"] == df['gap'].max()]
print(biggest_gap)
#the biggest gap occured in Mexico in April 2010. The gap was 239 degrees and the reported intensity of the 
#earthquake was 9, which is almost the highest reported intensity of the dataset. 
#the depth is 9.987, where the rupture began 
#depth = the depth where the earthquake begins to rupture


# In[37]:



df1 = df[(df['depth'] > 9.5) & (df['depth'] < 10.0)]
df1
#hier zien we dat de plek van de rupture geen invloed heeft op de intensity van de earthquake


# In[38]:


#CONCLUSIE: ESTIMATED/REPORTED INTENSITY VS. DEPTH -> GEEN VERBAND


# In[ ]:





# In[39]:


#LARGEST SIG 
largest_sig = df[df["sig"] == df['sig'].max()]
print(largest_sig)
#largest significant measured earthquake is also the one April 2010 in Mexico, but also another earthquake in 
#august 2017 Mexico, this one does had a tsunami. 


# In[40]:


fig = px.scatter(df, x="reported intensity", y="sig", color="year", color_discrete_sequence=["green", "yellow", "orange", "red"], title = 'Reported intensity vs. significance')
fig.show()


# In[41]:


#north america continent
#north_america = df['continent'].where('North America')
#show only north america earthqueakes over the years


# In[42]:


plt.figure(figsize=(15, 10))
sns.heatmap(df[['magnitude','nst','estimated intensity','sig','depth']].corr(), annot=True,linecolor = 'black', cmap='Blues')
plt.show()
#nst = the total number of seismic stations used to determine earthquake location


# In[43]:


fig = px.scatter(df, x="magnitude", y="sig", color="year", color_discrete_sequence=["green", "yellow", "orange", "red"], title = 'Magnitude vs. significance', trendline = 'ols')
fig.show()
#hoe hoger de magnitude, des te hoger de sig (wat ook wel logisch is)


# In[44]:


solar = solar[['earthquake.time', 'earthquake.latitude', 'earthquake.longitude', 'earthquake.mag', 'earthquake.place', 'MoonPhase.dynamic', 'MoonPhase.percent', 'day.duration', 'night.duration', 'Sun.height', 'Sun.speed', 'Moon.height', 'Moon.speed', 'Mars.height', 'Mars.speed']]


# In[45]:


solar.head()


# In[46]:


df.head()


# In[47]:


new_df = pd.merge(solar, df,  how='inner', left_on=['earthquake.latitude','earthquake.longitude'], right_on = ['latitude','longitude'])


# In[48]:


new_df.head()


# In[49]:


new_corr_matrix = new_df[['Sun.height', 'Sun.speed', 'Moon.height', 'Mars.height', 'Mars.speed','earthquake.time', 'magnitude', 'reported intensity', 'estimated intensity', 'sig', 'depth', 'year']]


# In[50]:


plt.figure(figsize=(20, 20))
sns.heatmap(new_df[['Sun.height', 'Sun.speed', 'Moon.height', 'Mars.height', 'Mars.speed', 'magnitude', 'reported intensity', 'estimated intensity', 'sig', 'depth', 'year']].corr(), annot=True,linecolor = 'black', cmap='Blues')
plt.show()
#mars.height and sun.height 0.36


# In[51]:


fig = px.scatter(new_df, x="magnitude", y="Sun.height", color = 'year', color_discrete_sequence=["green", "yellow", "orange", "red"], title = 'Magnitude vs. significance', trendline = 'ols'), 
fig.show()


# In[ ]:


#dropdown voor continent, scatterplot 


# In[ ]:





# In[ ]:


#slider/dropdown 1D


# In[52]:


import plotly.graph_objs as go
import plotly.express as px


# In[62]:


plt = px.box(new_df, x='tsunami', y='Moon.height', color='continent')

plt.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="All continents",
                     method="update",
                     args=[{"visible": [True] * len(plt.data)},
                           {"title": "All continents"}]),
                dict(label="Asia",
                     method="update",
                     args=[{"visible": [trace.name == 'Asia' for trace in plt.data]},
                           {"title": "Asia"}]),
                dict(label="South America",
                     method="update",
                     args=[{"visible": [trace.name == 'South America' for trace in plt.data]},
                           {"title": "South America"}]),
                dict(label="Europe",
                     method="update",
                     args=[{"visible": [trace.name == 'Europe' for trace in plt.data]},
                           {"title": "Europe"}]),
                dict(label="North America",
                     method="update",
                     args=[{"visible": [trace.name == 'North America' for trace in plt.data]},
                           {"title": "North America"}]),
                dict(label="Africa",
                     method="update",
                     args=[{"visible": [trace.name == 'Africa' for trace in plt.data]},
                           {"title": "Africa"}]),
            ]),
        )
    ])



plt.show()
#in deze plot zien we of de hoogte van de maan samenhangt met het ontstaan van een tsunami per continent. 
#bij de slider optie 'all continents' kunnen de continenten allemaal vergeleken worden.


# In[54]:


plt = px.box(new_df, x='MoonPhase.dynamic', y='magnitude', title = 'is the magnitude dependend on the phase of the moon?')
plt.show()


# In[89]:


plt = px.scatter(new_df, x='MoonPhase.percent', y='magnitude', animation_frame="year",trendline = 'ols')
plt.show()
#in deze plot zien we per jaar wat de magnitude bij een percentage maanfase is
#ook kunnen we meteen de hoeveelheid aardbevingen per jaar zien wereldwijd
#de trendlijn laat zien of er een verband is tussen hoogte van de magnitude en hoogte van het maan percentage
#moonphase.percent laat zien welk percentage van de maan zichtbaar is vanuit de aarde


# In[90]:


plt = px.scatter(new_df, x='sig', y='magnitude', animation_frame="year",trendline = 'ols')
plt.show()
#hier zien we een duidelijk verschil tussen significantie van de aardbeving en de magnitude. 
#de trendlijn zal van 2001 steeds minder steil stijgen naar aanloop van 2016


# In[ ]:




