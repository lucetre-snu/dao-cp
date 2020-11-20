- Made by Dawon Ahn
- Name : MadridAirQaulity
- Source : https://www.kaggle.com/decide-soluciones/air-quality-madrid#madrid_2006.csv 
- Citation:

- Mode : 3
- Format : date, location, air pollutants, (measurement)
	1) 1hour_airquality 
	2) 6hours_airquality 
	3) 1day_airquality 
- Dimension :
	1) 1hour_airquality : 64,248 * 24 * 14
	2) 6hours_airquality :  10,709 * 24 * 14
	3) 1day_airquality : 2,678 * 24 * 14
- Nonzeros :
	1) 1hour_airquality : 8,036,759
	2) 6hours_airquality : 1,346,745 
	3) 1day_airquality : 337,759 

- Description : 
* Air Quality in Madrid recorded during 7 years from 2011/01/01 to 2018/05
* Preprocessed : Original data recoreded hourly. 6 hours and 1 day indicates a window size to average value so that each represents air quality data recorded by 6 hours and daily, respectively. 
* Including Missing data.
* COO format (e.g. i j k v for 4 mode tensor haivng i, j, k indices and v value).
* 0-based indexing.
* date, location, pollutants dictionaries for real values.
* original_data directory includes raw data directly downloaded from a source before making a tensor.



