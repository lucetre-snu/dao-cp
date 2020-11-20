- Made by Dawon Ahn
- Name : KoreaAirQaulity
- Source : https://www.airkorea.or.kr/eng
- Citation:

- Mode : 3
- Format :  date, location, air pollutants, (measurement)
	1) seoul_airquality.tensor 
	2) korea_airquality.tensor 
- Dimension :
	1) seoul : 9,478 * 37 * 6 
	2) korea : 9,478 * 323 * 6 
- Nonzeros :
	1) seoul : 1,632,264
	2) korea : 11,270,028

- Description : 
* Air Quality in Seoul and Korea recorded daily from 2018/09/01 to 2019/09/31.
* Including Missing data 
* COO format (e.g. i j k v for 4 mode tensor haivng i, j, k indices and v value)
* 0-based indexing
* date , location, pollutants dictionaries for real values
* original_data directory includes raw data directly downloaded from a source before making a tensor.



