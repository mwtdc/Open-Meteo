#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import logging
import os
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import numpy as np
import pandas as pd
import pymysql
import pyodbc
import requests
import xgboost as xgb
import yaml
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

# 98, 277 строка - указать количество дней,
# на которых готовить прогноз (база корректно наполнена с 28.07.2022,
# до этого с 14.06.2022 загружено за один день)

start_time = datetime.datetime.now()
warnings.filterwarnings('ignore')

print('*********** Open_Meteo Start!!! **********', datetime.datetime.now())

# Общий раздел

# Настройки для логера
if platform == "linux" or platform == "linux2":
    logging.basicConfig(filename="/var/log/log-execute/log_journal_openmeteo_rsv.log.txt",
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")
elif platform == "win32":
    logging.basicConfig(filename=f'{pathlib.Path(__file__).parent.absolute()}/log_journal_openmeteo_rsv.log.txt',
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")
#

# Загружаем yaml файл с настройками
with open(f'{pathlib.Path(__file__).parent.absolute()}/settings.yaml', 'r') as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings['telegram'])
sql_settings = pd.DataFrame(settings['sql_db'])
pyodbc_settings = pd.DataFrame(settings['pyodbc_db'])
#

# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)


def telegram(i, text):
    msg = urllib.parse.quote(str(text))
    bot_token = str(telegram_settings.bot_token[i])
    channel_id = str(telegram_settings.channel_id[i])

    retry_strategy = Retry(
        total=3,
        status_forcelist=[101, 429, 500, 502, 503, 504],
        method_whitelist=["GET", "POST"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    http.post(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}', timeout=10)
#

# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)


def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    return pymysql.connect(host=host_yaml,
                           user=user_yaml,
                           port=port_yaml,
                           password=password_yaml,
                           database=database_yaml)
#

# Функция загрузки факта выработки
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)


def fact_load(i):
    server = str(pyodbc_settings.host[i]) 
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i]) 
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == "linux" or platform == "linux2":
        connection_ms = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID='+username+';PWD=' + password)
    elif platform == "win32":
        connection_ms = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute("SELECT SUBSTRING (Points.PointName ,len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT, SUM(Val) as Val FROM Points JOIN PointParams ON Points.ID_Point=PointParams.ID_Point JOIN PointMains ON PointParams.ID_PP=PointMains.ID_PP WHERE PointName like 'Генерация%{G%' AND ID_Param=153 AND DT >= DATEADD(HOUR, -70 * 24, DATEDIFF(d, 0, GETDATE())) AND PointName NOT LIKE '%GVIE0001%' AND PointName NOT LIKE '%GVIE0012%' AND PointName NOT LIKE '%GVIE0416%' AND PointName NOT LIKE '%GVIE0167%' AND PointName NOT LIKE '%GVIE0264%' AND PointName NOT LIKE '%GVIE0007%' AND PointName NOT LIKE '%GVIE0680%' AND PointName NOT LIKE '%GVIE0987%' AND PointName NOT LIKE '%GVIE0988%' AND PointName NOT LIKE '%GVIE0989%' AND PointName NOT LIKE '%GVIE0991%' AND PointName NOT LIKE '%GVIE0994%' AND PointName NOT LIKE '%GVIE1372%' GROUP BY SUBSTRING (Points.PointName ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT), DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT) ORDER BY SUBSTRING (Points.PointName ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT), DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT);")
    #######################################################################################################################################################################################################################################################################################################################################
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=['gtp', 'dt', 'fact'])
    fact.drop_duplicates(subset=['gtp', 'dt'],
                         keep='last', inplace=True,
                         ignore_index=False)
    return fact
#

# Раздел загрузки прогноза погоды в базу


telegram(1, "Open_Meteo: Старт загрузки погоды.")

# Задаем переменные (даты для прогноза и список погодных параметров)
date_beg = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
date_end = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
weather_dataframe = pd.DataFrame()
col_parameters = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                  'apparent_temperature', 'pressure_msl', 'surface_pressure',
                  'cloudcover', 'cloudcover_low', 'cloudcover_mid',
                  'cloudcover_high', 'windspeed_10m', 'windspeed_80m',
                  'windspeed_120m', 'windspeed_180m', 'winddirection_10m',
                  'winddirection_80m', 'winddirection_120m',
                  'winddirection_180m', 'windgusts_10m', 'shortwave_radiation',
                  'direct_radiation', 'direct_normal_irradiance',
                  'diffuse_radiation', 'vapor_pressure_deficit',
                  'evapotranspiration', 'et0_fao_evapotranspiration',
                  'precipitation', 'snowfall', 'rain', 'showers', 'weathercode',
                  'snow_depth', 'freezinglevel_height', 'soil_temperature_0cm',
                  'soil_temperature_6cm', 'soil_temperature_18cm',
                  'soil_temperature_54cm', 'soil_moisture_0_1cm',
                  'soil_moisture_1_3cm', 'soil_moisture_3_9cm',
                  'soil_moisture_9_27cm', 'soil_moisture_27_81cm']

list_parameters = (','.join(col_parameters))
#

# Загрузка списка ГТП и координат из базы
connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = "select gtp,lat,lng from visualcrossing.ses_gtp where gtp not like 'GK%';"
    cursor.execute(sql)
    ses_dataframe = pd.DataFrame(cursor.fetchall(), columns=['gtp', 'lat', 'lng'])
    connection_geo.close()
logging.info(f'Список ГТП и координаты станций загружены из базы visualcrossing.ses_gtp')
#

# Загрузка прогнозов погоды по станциям
g = 0
for ses in range(len(ses_dataframe.index)):
    gtp = str(ses_dataframe.gtp[ses])
    lat = str(ses_dataframe.lat[ses]).replace(',', '.')
    lng = str(ses_dataframe.lng[ses]).replace(',', '.')
    print(gtp)
    try:
        url_response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly={list_parameters}')
        url_response.raise_for_status()
        while url_response.status_code !=200:
            url_response = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly={list_parameters}')
            sleep(20)
        if url_response.ok:
            json_string = json.loads(url_response.text)
            json_req_pd = pd.DataFrame(data=json_string['hourly'])
            json_req_pd.insert(1, 'gtp', gtp)
            json_req_pd.insert(2, 'datetime_msc', (pd.to_datetime(json_req_pd['time'], utc=False) + pd.DateOffset(hours=3)))
            json_req_pd.drop(['time'], axis='columns', inplace=True)
            # print(dataframe_1)
            weather_dataframe = weather_dataframe.append(json_req_pd,
                                                         ignore_index=True)
            g += 1
            print("прогноз погоды загружен")
            logging.info(str(g)+" Прогноз погоды для ГТП "+str(gtp)+" загружен с open-meteo.com")
        else:
            print('Open_Meteo: Ошибка запроса:' + str(url_response.status_code))
            logging.error('Open_Meteo: Ошибка запроса:' + str(url_response.status_code))
            telegram(1, 'Open_Meteo: Ошибка запроса:' + str(url_response.status_code))
            os._exit(1)
    except requests.HTTPError as http_err:
        print(f'Open_Meteo: HTTP error occurred: {http_err.response.text}')
        logging.error(f'Open_Meteo: HTTP error occurred: {http_err.response.text}')
        telegram(1, f'Open_Meteo: HTTP error occurred: {http_err.response.text}')
        os._exit(1)
    except Exception as err:
        print(f'Open_Meteo: Other error occurred: {err}')
        logging.error(f'Open_Meteo: Other error occurred: {err}')
        telegram(1, f'Open_Meteo: Other error occurred: {err}')
        os._exit(1)
weather_dataframe.fillna(0, inplace=True)
# weather_dataframe.to_excel(f'{pathlib.Path(__file__).parent.absolute()}/weather_dataframe.xlsx')   
# print(weather_dataframe)
telegram(1, f'Open_Meteo: загружен прогноз для {g} гтп')
logging.info(f'Сформирован датафрейм для {g} гтп')
#

col_to_database = ['gtp', 'datetime_msc', 'loadtime', 'temperature_2m',
                   'relativehumidity_2m', 'dewpoint_2m', 'apparent_temperature',
                   'pressure_msl', 'surface_pressure', 'cloudcover',
                   'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
                   'windspeed_10m', 'windspeed_80m', 'windspeed_120m',
                   'windspeed_180m', 'winddirection_10m', 'winddirection_80m',
                   'winddirection_120m', 'winddirection_180m', 'windgusts_10m',
                   'shortwave_radiation', 'direct_radiation',
                   'direct_normal_irradiance', 'diffuse_radiation',
                   'vapor_pressure_deficit', 'evapotranspiration',
                   'et0_fao_evapotranspiration', 'precipitation', 'snowfall',
                   'rain', 'showers', 'weathercode', 'snow_depth',
                   'freezinglevel_height', 'soil_temperature_0cm',
                   'soil_temperature_6cm', 'soil_temperature_18cm',
                   'soil_temperature_54cm', 'soil_moisture_0_1cm',
                   'soil_moisture_1_3cm', 'soil_moisture_3_9cm',
                   'soil_moisture_9_27cm', 'soil_moisture_27_81cm']

list_col_database = (','.join(col_to_database))

connection_om = connection(0)
conn_cursor = connection_om.cursor()

vall = ''
rows = len(weather_dataframe.index)
gtp_rows = int(round(rows/168, 0))
for r in range(len(weather_dataframe.index)):
    vall = (vall+"('"
    + str(weather_dataframe.gtp[r])+"','"
    + str(weather_dataframe.datetime_msc[r])+"','"
    + str(datetime.datetime.now().isoformat())+"','"
    + str(weather_dataframe.temperature_2m[r])+"','"
    + str(weather_dataframe.relativehumidity_2m[r])+"','"
    + str(weather_dataframe.dewpoint_2m[r])+"','"
    + str(weather_dataframe.apparent_temperature[r])+"','"
    + str(weather_dataframe.pressure_msl[r])+"','"
    + str(weather_dataframe.surface_pressure[r])+"','"
    + str(weather_dataframe.cloudcover[r])+"','"
    + str(weather_dataframe.cloudcover_low[r])+"','"
    + str(weather_dataframe.cloudcover_mid[r])+"','"
    + str(weather_dataframe.cloudcover_high[r])+"','"
    + str(weather_dataframe.windspeed_10m[r])+"','"
    + str(weather_dataframe.windspeed_80m[r])+"','"
    + str(weather_dataframe.windspeed_120m[r])+"','"
    + str(weather_dataframe.windspeed_180m[r])+"','"
    + str(weather_dataframe.winddirection_10m[r])+"','"
    + str(weather_dataframe.winddirection_80m[r])+"','"
    + str(weather_dataframe.winddirection_120m[r])+"','"
    + str(weather_dataframe.winddirection_180m[r])+"','"
    + str(weather_dataframe.windgusts_10m[r])+"','"
    + str(weather_dataframe.shortwave_radiation[r])+"','"
    + str(weather_dataframe.direct_radiation[r])+"','"
    + str(weather_dataframe.direct_normal_irradiance[r])+"','"   
    + str(weather_dataframe.diffuse_radiation[r])+"','"
    + str(weather_dataframe.vapor_pressure_deficit[r])+"','"
    + str(weather_dataframe.evapotranspiration[r])+"','"
    + str(weather_dataframe.et0_fao_evapotranspiration[r])+"','"
    + str(weather_dataframe.precipitation[r])+"','"
    + str(weather_dataframe.snowfall[r])+"','"
    + str(weather_dataframe.rain[r])+"','"
    + str(weather_dataframe.showers[r])+"','"
    + str(weather_dataframe.weathercode[r])+"','"
    + str(weather_dataframe.snow_depth[r])+"','"
    + str(weather_dataframe.freezinglevel_height[r])+"','"
    + str(weather_dataframe.soil_temperature_0cm[r])+"','"
    + str(weather_dataframe.soil_temperature_6cm[r])+"','"
    + str(weather_dataframe.soil_temperature_18cm[r])+"','"
    + str(weather_dataframe.soil_temperature_54cm[r])+"','"
    + str(weather_dataframe.soil_moisture_0_1cm[r])+"','"
    + str(weather_dataframe.soil_moisture_1_3cm[r])+"','"
    + str(weather_dataframe.soil_moisture_3_9cm[r])+"','"
    + str(weather_dataframe.soil_moisture_9_27cm[r])+"','"
    + str(weather_dataframe.soil_moisture_27_81cm[r])+"'"+'),')

vall = vall[:-1]
sql = (f'INSERT INTO visualcrossing.openmeteo ({list_col_database}) VALUES {vall};')
conn_cursor.execute(sql)
connection_om.commit()
connection_om.close()

# Уведомление о записи в БД
telegram(0, f'Open_Meteo: записано в БД {rows} строк ({gtp_rows} гтп)')
logging.info(f'записано в БД {rows} строк прогноза погоды ({gtp_rows} гтп)')
#

# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов

col_to_database = ['gtp', 'datetime_msc', 'loadtime', 'temperature_2m',
                   'relativehumidity_2m', 'dewpoint_2m', 'apparent_temperature',
                   'pressure_msl', 'surface_pressure', 'cloudcover',
                   'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
                   'windspeed_10m', 'windspeed_80m', 'windspeed_120m',
                   'windspeed_180m', 'winddirection_10m', 'winddirection_80m',
                   'winddirection_120m', 'winddirection_180m', 'windgusts_10m',
                   'shortwave_radiation', 'direct_radiation',
                   'direct_normal_irradiance', 'diffuse_radiation',
                   'vapor_pressure_deficit', 'evapotranspiration',
                   'et0_fao_evapotranspiration', 'precipitation', 'snowfall',
                   'rain', 'showers', 'weathercode', 'snow_depth',
                   'freezinglevel_height', 'soil_temperature_0cm',
                   'soil_temperature_6cm', 'soil_temperature_18cm',
                   'soil_temperature_54cm', 'soil_moisture_0_1cm',
                   'soil_moisture_1_3cm', 'soil_moisture_3_9cm',
                   'soil_moisture_9_27cm', 'soil_moisture_27_81cm']

list_col_database = (','.join(col_to_database))

connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = f'select gtp,def_power from visualcrossing.ses_gtp;'
    cursor.execute(sql)
    ses_dataframe = pd.DataFrame(cursor.fetchall(), columns=['gtp', 'def_power'])
    ses_dataframe['def_power']=ses_dataframe['def_power']*1000
    # ses_dataframe=ses_dataframe[ses_dataframe['gtp'].str.contains('GVIE', regex=False)]
    ses_dataframe = ses_dataframe[(ses_dataframe['gtp'].str.contains('GVIE', regex=False)) | 
                                  (ses_dataframe['gtp'].str.contains('GKZ', regex=False)) | 
                                  (ses_dataframe['gtp'].str.contains('GROZ', regex=False))]
    connection_geo.close()
# print(ses_dataframe)
connection_forecast = connection(0)
with connection_forecast.cursor() as cursor:
    sql = f'select {list_col_database} from visualcrossing.openmeteo where loadtime >= CURDATE() - INTERVAL 70 DAY;'
    cursor.execute(sql)
    forecast_dataframe = pd.DataFrame(cursor.fetchall(),
                                      columns=col_to_database)
    connection_forecast.close()
logging.info("Загружен массив прогноза погоды за предыдущие дни")
# Удаление дубликатов прогноза, т.к.
# каждый день грузит на 7 дней вперед и получается накладка
date_beg_predict = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
date_end_predict = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
forecast_dataframe.drop_duplicates(subset=['datetime_msc', 'gtp'],
                                   keep='last',
                                   inplace=True,
                                   ignore_index=False)
forecast_dataframe['month'] = pd.to_datetime(forecast_dataframe.datetime_msc.values).month
forecast_dataframe['hour'] = pd.to_datetime(forecast_dataframe.datetime_msc.values).hour

test_dataframe = forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] < str(date_beg_predict))[0]])
test_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] > str(date_end_predict))[0]], inplace=True)

# Строка ниже - из датафрейма для предикта оставляем
# только гтп начинающиеся не с GK
test_dataframe = test_dataframe[(test_dataframe['gtp'].str.contains('GVIE', regex=False)) | 
                                (test_dataframe['gtp'].str.contains('GROZ', regex=False))]
test_dataframe = test_dataframe.merge(ses_dataframe,
                                      left_on=['gtp'],
                                      right_on=['gtp'],
                                      how='left')

forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] > str(datetime.datetime.today()))[0]], inplace=True)
# Сортировка датафрейма по гтп и дате
forecast_dataframe.sort_values(['gtp', 'datetime_msc'], inplace=True)
forecast_dataframe['datetime_msc'] = forecast_dataframe['datetime_msc'].astype('datetime64[ns]')
logging.info("forecast_dataframe и test_dataframe преобразованы в нужный вид")

#
fact = fact_load(0)
#
# print(fact)

forecast_dataframe = forecast_dataframe.merge(ses_dataframe,
                                              left_on=['gtp'],
                                              right_on=['gtp'],
                                              how='left')
forecast_dataframe = forecast_dataframe.merge(fact,
                                              left_on=['gtp', 'datetime_msc'],
                                              right_on=['gtp', 'dt'],
                                              how='left')
# print(forecast_dataframe)
# forecast_dataframe.to_excel('forecast_dataframe.xlsx')

forecast_dataframe.dropna(subset=['fact'], inplace=True)
forecast_dataframe.drop(['dt', 'loadtime'], axis='columns', inplace=True)
# print(forecast_dataframe)
# print(test_dataframe)
# forecast_dataframe.to_excel("forecast_dataframe.xlsx")
# test_dataframe.to_excel("test_dataframe.xlsx")

col_to_float = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                'apparent_temperature', 'pressure_msl', 'surface_pressure',
                'cloudcover', 'cloudcover_low', 'cloudcover_mid',
                'cloudcover_high', 'windspeed_10m', 'windspeed_80m',
                'windspeed_120m', 'windspeed_180m', 'winddirection_10m',
                'winddirection_80m', 'winddirection_120m', 'winddirection_180m',
                'windgusts_10m', 'shortwave_radiation', 'direct_radiation',
                'direct_normal_irradiance', 'diffuse_radiation',
                'vapor_pressure_deficit', 'evapotranspiration',
                'et0_fao_evapotranspiration', 'precipitation', 'snowfall',
                'rain', 'showers', 'weathercode', 'snow_depth',
                'freezinglevel_height', 'soil_temperature_0cm',
                'soil_temperature_6cm', 'soil_temperature_18cm',
                'soil_temperature_54cm', 'soil_moisture_0_1cm',
                'soil_moisture_1_3cm', 'soil_moisture_3_9cm',
                'soil_moisture_9_27cm', 'soil_moisture_27_81cm', 'def_power']
for col in col_to_float:
    forecast_dataframe[col] = forecast_dataframe[col].astype('float')
    test_dataframe[col] = test_dataframe[col].astype('float')

col_to_int = ['month','hour']
for col in col_to_int:
    forecast_dataframe[col] = forecast_dataframe[col].astype('int')

logging.info("Датафреймы погоды и факта выработки склеены")

# Раздел подготовки прогноза на XGBoost
# XGBoost
z = forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['fact'] == 0)])
z['gtp'] = z['gtp'].str.replace('GVIE','1')
z['gtp'] = z['gtp'].str.replace('GKZV','4')
z['gtp'] = z['gtp'].str.replace('GKZ','2')
z['gtp'] = z['gtp'].str.replace('GROZ','3')
x = z.drop(['fact','datetime_msc'], axis = 1)
# print(x)
# x.to_excel("x.xlsx")
y = z['fact']

predict_dataframe = test_dataframe.drop(['datetime_msc','loadtime'], axis = 1)
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GVIE','1')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV','4')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZ','2')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GROZ','3')
# print(predict_dataframe)
# predict_dataframe.to_excel("predict_dataframe.xlsx")
x['gtp'] = x['gtp'].astype('int')
predict_dataframe['gtp'] = predict_dataframe['gtp'].astype('int')
#
x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8)
logging.info("Старт предикта на XGBoostRegressor")

param = {
    'lambda': 0.10427064120338686,
    'alpha': 0.0023793948424012125,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'learning_rate': 0.014,
    'n_estimators': 10000,
    'max_depth': 11,
    'random_state': 2020,
    'min_child_weight': 1,
}
reg = xgb.XGBRegressor(**param)
regr = BaggingRegressor(base_estimator=reg,n_estimators=3,n_jobs=-1).fit(x_train, y_train)
predict = regr.predict(predict_dataframe)
test_dataframe['forecast'] = pd.DataFrame(predict)

logging.info("Подготовлен прогноз на CatBoostRegressor")
#
# Обработка прогнозных значений
# Обрезаем по максимум за месяц в часы
max_month_dataframe = pd.DataFrame()
date_cut = (datetime.datetime.today() + datetime.timedelta(days=-29)).strftime("%Y-%m-%d")
cut_dataframe = forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] < str(date_cut))[0]])
for gtp in test_dataframe.gtp.value_counts().index:
    max_month=cut_dataframe.loc[cut_dataframe.gtp==gtp,['fact', 'hour', 'gtp']].groupby(by=['hour']).max()
    max_month_dataframe = max_month_dataframe.append(max_month, ignore_index=True)
max_month_dataframe['hour'] = test_dataframe['hour']
test_dataframe = test_dataframe.merge(max_month_dataframe, left_on=['gtp', 'hour'], right_on = ['gtp', 'hour'], how='left')
test_dataframe['forecast'] = test_dataframe[['forecast', 'fact', 'def_power']].min(axis=1)
# Если прогноз отрицательный, то 0
test_dataframe.forecast[test_dataframe.forecast<0] = 0
test_dataframe.drop(['fact', 'month', 'hour', 'loadtime'], axis='columns', inplace=True)
test_dataframe.drop(col_to_float, axis='columns', inplace=True)
test_dataframe.to_excel(f'{pathlib.Path(__file__).parent.absolute()}/{(datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%d.%m.%Y")}_xgboost_2day_all.xlsx')
#test_dataframe.to_excel(f'{pathlib.Path(__file__).parent.absolute()}/xgboost_2day_all.xlsx')
logging.info("Датафрейм с прогнозом выработки прошел обработку от нулевых значений и обрезку по макс за месяц")
#

#
# Запись прогноза в БД
connection_vc = connection(2)
conn_cursor = connection_vc.cursor()
vall_predict = ''
for p in range(len(test_dataframe.index)):
    vall_predict = (vall_predict + "('"
    + str(test_dataframe.gtp[p]) + "','"
    + str(test_dataframe.datetime_msc[p]) + "','"
    + "20" + "','"                      
    + str(datetime.datetime.now().isoformat()) + "','"                       
    + str(round(test_dataframe.forecast[p],3)) + "'" + '),')  
vall_predict = vall_predict[:-1]
sql_predict = ("INSERT INTO weather_foreca (gtp,dt,id_foreca,load_time,value) VALUES " + vall_predict + ";")
# print(sql_predict)
conn_cursor.execute(sql_predict) 
connection_vc.commit()
connection_vc.close()
#
# Уведомление о подготовке прогноза
telegram(0,"Open_Meteo: прогноз подготовлен")
logging.info("Прогноз записан в БД treid_03")
print('Время выполнения:', datetime.datetime.now() - start_time)
