Attribute Information after Preprocessing:

| Attribute | Stands for |
|:---|:---|
|Year|Year of data in this row|
|Month|Month of data in this row|
|Day|Day of data in this row|
|Hour|Hour of data in this row|
|Season|Season of data in this row|
|DEWP|Dew Point (Celsius Degree)|
|TEMP|Temperature (Celsius Degree)|
|HUMI|Humidity (%)|
|PRES|Pressure (hPa)|
|CBWD|Combined wind direction NE = 0, NW = 1, SE = 2, SW = 3, cv = 4|
|IWS|Cumulated wind speed (m/s)|
|RAIN|hourly precipitation (mm)|
|IPREC|Cumulated precipitation (mm)|
|City|Bejing = 0, Chengdu = 1, Guangzhou = 2, Shanghai = 3, Shenyang = 4|
|AVG PM2.5|Average PM2.5 concentration (ug/m^3)|

Data Preprocessing:  
1. Remove records which don't have PM2.5 data from every observatory  
2. Remove records which don't have data of any of weather information
3. Remove records which contain outlier values of any feature
4. Calculate average PM2.5 from each observatory with PM2.5 data  
5. Transform CBWD and City to numerical data
6. Normalize continuous features to between 0 to 1

[PM2.5 Data](https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities)