#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import logging
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import numpy as np
import optuna
import pandas as pd
import pyodbc
import requests
import xgboost as xgb
import yaml
from optuna.samplers import TPESampler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Коэффициент завышения прогноза:
OVERVALUE_COEFF = 1.02
# Задаем переменные (даты для прогноза и список погодных параметров)
DATE_BEG = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime(
    "%Y-%m-%d"
)
DATE_END = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime(
    "%Y-%m-%d"
)
COL_PARAMETERS = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "windspeed_10m",
    "windspeed_80m",
    "windspeed_120m",
    "windspeed_180m",
    "winddirection_10m",
    "winddirection_80m",
    "winddirection_120m",
    "winddirection_180m",
    "windgusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "vapor_pressure_deficit",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "precipitation",
    "snowfall",
    "rain",
    "showers",
    "weathercode",
    "snow_depth",
    "freezinglevel_height",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_1cm",
    "soil_moisture_1_3cm",
    "soil_moisture_3_9cm",
    "soil_moisture_9_27cm",
    "soil_moisture_27_81cm",
]


start_time = datetime.datetime.now()
warnings.filterwarnings("ignore")

print("Open_Meteo Start!", datetime.datetime.now())


# Настройки для логера
if platform == "linux" or platform == "linux2":
    logging.basicConfig(
        filename="/var/log/log-execute/log_journal_openmeteo_rsv.log.txt",
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d -"
            " %(message)s"
        ),
    )
elif platform == "win32":
    logging.basicConfig(
        filename=f"{pathlib.Path(__file__).parent.absolute()}/log_journal_openmeteo_rsv.log.txt",
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d -"
            " %(message)s"
        ),
    )

# Загружаем yaml файл с настройками
with open(
    f"{pathlib.Path(__file__).parent.absolute()}/settings.yaml", "r"
) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings["telegram"])
sql_settings = pd.DataFrame(settings["sql_db"])
pyodbc_settings = pd.DataFrame(settings["pyodbc_db"])


# Функция отправки уведомлений в telegram на любое количество каналов
#  (указать данные в yaml файле настроек)
def telegram(i, text):
    try:
        msg = urllib.parse.quote(str(text))
        bot_token = str(telegram_settings.bot_token[i])
        channel_id = str(telegram_settings.channel_id[i])

        retry_strategy = Retry(
            total=3,
            status_forcelist=[101, 429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        http.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}",
            verify=False,
            timeout=10,
        )
    except Exception as err:
        print(f"Openmeteo_KZ: Ошибка при отправке в telegram -  {err}")
        logging.error(f"Openmeteo_KZ: Ошибка при отправке в telegram -  {err}")


# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    db_data = f"mysql://{user_yaml}:{password_yaml}@{host_yaml}:{port_yaml}/{database_yaml}"
    return create_engine(db_data).connect()


# Функция загрузки факта выработки
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def fact_load(i, dt):
    server = str(pyodbc_settings.host[i])
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i])
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == "linux" or platform == "linux2":
        connection_ms = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    elif platform == "win32":
        connection_ms = pyodbc.connect(
            "DRIVER={SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute(
        "SELECT SUBSTRING (Points.PointName ,"
        "len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT,"
        " SUM(Val) as Val FROM Points JOIN PointParams ON "
        "Points.ID_Point=PointParams.ID_Point JOIN PointMains"
        " ON PointParams.ID_PP=PointMains.ID_PP WHERE "
        "PointName like 'Генерация%{G%' AND ID_Param=153 "
        "AND DT >= "
        + str(dt)
        + " AND PointName NOT LIKE '%GVIE0001%' AND"
        " PointName NOT LIKE '%GVIE0012%' AND PointName NOT LIKE '%GVIE0416%'"
        " AND PointName NOT LIKE '%GVIE0167%' AND PointName NOT LIKE"
        " '%GVIE0007%' AND PointName NOT LIKE '%GVIE0987%' AND PointName NOT"
        " LIKE '%GVIE0988%' AND PointName NOT LIKE '%GVIE0989%' AND PointName"
        " NOT LIKE '%GVIE0991%' AND PointName NOT LIKE '%GVIE0994%' AND"
        " PointName NOT LIKE '%GVIE1372%' AND PointName NOT LIKE '%GKZV0013%'"
        " GROUP BY SUBSTRING (Points.PointName"
        " ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT), DATEPART(MONTH,"
        " DT), DATEPART(DAY, DT), DATEPART(HOUR, DT) ORDER BY SUBSTRING"
        " (Points.PointName ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT),"
        " DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT);"
    )
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=["gtp", "dt", "fact"])
    fact.drop_duplicates(
        subset=["gtp", "dt"], keep="last", inplace=True, ignore_index=False
    )
    fact["fact"] = fact["fact"].astype("float").round(-2)
    return fact


# Функция записи датафрейма в базу
def load_data_to_db(db_name, connect_id, dataframe):
    telegram(1, "Openmeteo_KZ: Старт записи в БД.")
    logging.info("Openmeteo_KZ: Старт записи в БД.")

    dataframe = pd.DataFrame(dataframe)
    connection_skm = connection(connect_id)
    try:
        dataframe.to_sql(
            name=db_name,
            con=connection_skm,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        rows = len(dataframe)
        telegram(
            1,
            f"Openmeteo_KZ: записано в БД {rows} строк ({int(rows/48)} гтп)",
        )
        if len(dataframe.columns) > 30:
            telegram(
                0,
                f"Openmeteo_KZ: записано в БД {rows} строк"
                f" ({int(rows/48)} гтп)",
            )
        logging.info(
            f"записано в БД {rows} строк c погодой ({int(rows/48)} гтп)"
        )
        telegram(1, "Openmeteo_KZ: Финиш записи в БД.")
        logging.info("Openmeteo_KZ: Финиш записи в БД.")
    except Exception as err:
        telegram(1, f"Openmeteo_KZ: Ошибка записи в БД: {err}")
        logging.info(f"Openmeteo_KZ: Ошибка записи в БД: {err}")


# Функция загрузки датафрейма из базы
def load_data_from_db(
    db_name,
    col_from_database,
    connect_id,
    condition_column,
    day_interval,
):
    telegram(1, "Openmeteo_KZ: Старт загрузки из БД.")
    logging.info("Openmeteo_KZ: Старт загрузки из БД.")

    list_col_database = ",".join(col_from_database)
    connection_db = connection(connect_id)
    if day_interval is None and condition_column is None:
        query = f"select {list_col_database} from {db_name};"
    else:
        query = (
            f"select {list_col_database} from {db_name} where"
            f" {condition_column} >= CURDATE() - INTERVAL {day_interval} DAY;"
        )
    dataframe_from_db = pd.read_sql(sql=query, con=connection_db)

    telegram(1, "Openmeteo_KZ: Финиш загрузки из БД.")
    logging.info("Openmeteo_KZ: Финиш загрузки из БД.")
    return dataframe_from_db


# Раздел загрузки прогноза погоды в базу
def load_forecast_to_db(date_beg, date_end, col_parameters):
    telegram(1, "Open_Meteo: Старт загрузки погоды.")

    weather_dataframe = pd.DataFrame()
    list_parameters = ",".join(col_parameters)

    # Загрузка списка ГТП и координат из базы
    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "lat", "lng"],
        0,
        None,
        None,
    )

    # Ниже можно выбирать гтп в датафрейме, только опт, кз, розн или все.
    # ses_dataframe = ses_dataframe[
    #     ses_dataframe["gtp"].str.contains("GK", regex=False)
    # ]
    ses_dataframe = ses_dataframe[
        (ses_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (ses_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    ses_dataframe.reset_index(inplace=True)

    logging.info(
        f"Список ГТП и координаты станций загружены из базы"
        f" visualcrossing.ses_gtp"
    )

    # Загрузка прогнозов погоды по станциям
    g = 0
    for ses in range(len(ses_dataframe.index)):
        gtp = str(ses_dataframe.gtp[ses])
        lat = str(ses_dataframe.lat[ses]).replace(",", ".")
        lng = str(ses_dataframe.lng[ses]).replace(",", ".")
        print(gtp)
        try:
            url_response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly={list_parameters}",
                verify=False,
            )
            # url_response.raise_for_status()
            while url_response.status_code != 200:
                url_response = requests.get(
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly={list_parameters}",
                    verify=False,
                )
                sleep(60)
            if url_response.ok:
                json_string = json.loads(url_response.text)
                json_req_pd = pd.DataFrame(data=json_string["hourly"])
                json_req_pd.insert(1, "gtp", gtp)
                json_req_pd.insert(
                    2,
                    "datetime_msc",
                    (
                        pd.to_datetime(json_req_pd["time"], utc=False)
                        + pd.DateOffset(hours=3)
                    ),
                )
                json_req_pd.insert(
                    3, "loadtime", datetime.datetime.now().isoformat()
                )
                json_req_pd.drop(["time"], axis="columns", inplace=True)
                # print(dataframe_1)
                weather_dataframe = weather_dataframe.append(
                    json_req_pd, ignore_index=True
                )
                g += 1
                print("прогноз погоды загружен")
                logging.info(
                    f"{g} Прогноз погоды для ГТП {gtp} загружен с"
                    " open-meteo.com"
                )
            else:
                print(
                    f"Open_Meteo: Ошибка запроса: {url_response.status_code}"
                )
                logging.error(
                    f"Open_Meteo: Ошибка запроса: {url_response.status_code}"
                )
                telegram(
                    1,
                    f"Open_Meteo: Ошибка запроса: {url_response.status_code}",
                )
                # os._exit(1)
        except requests.HTTPError as http_err:
            print(f"Open_Meteo: HTTP error occurred: {http_err.response.text}")
            logging.error(
                f"Open_Meteo: HTTP error occurred: {http_err.response.text}"
            )
            telegram(
                1,
                f"Open_Meteo: HTTP error occurred: {http_err.response.text}",
            )
            # os._exit(1)
        except Exception as err:
            print(f"Open_Meteo: Other error occurred: {err}")
            logging.error(f"Open_Meteo: Other error occurred: {err}")
            telegram(1, f"Open_Meteo: Other error occurred: {err}")
            # os._exit(1)

    weather_dataframe.fillna(0, inplace=True)
    # weather_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/weather_dataframe.xlsx"
    # )
    # print(weather_dataframe)
    telegram(1, f"Open_Meteo: загружен прогноз для {g} гтп")
    logging.info(f"Сформирован датафрейм для {g} гтп")

    load_data_to_db(
        "openmeteo",
        0,
        weather_dataframe,
    )


# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов
def prepare_datasets_to_train():
    col_in_database = ["gtp", "datetime_msc", "loadtime"] + COL_PARAMETERS
    list_parameters = ",".join(col_in_database)

    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "def_power"],
        0,
        None,
        None,
    )
    ses_dataframe["def_power"] = ses_dataframe["def_power"] * 1000
    # ses_dataframe = ses_dataframe[
    #     ses_dataframe["gtp"].str.contains("GVIE", regex=False)
    # ]
    ses_dataframe = ses_dataframe[
        (ses_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (ses_dataframe["gtp"].str.contains("GKZ0", regex=False))
        | (ses_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    logging.info("Загружен датафрейм с гтп и установленной мощностью.")
    # print(ses_dataframe)

    forecast_dataframe = load_data_from_db(
        "visualcrossing.openmeteo",
        col_in_database,
        0,
        "loadtime",
        365,
    )
    logging.info("Загружен массив прогноза погоды за предыдущие дни")

    # Удаление дубликатов прогноза, т.к.
    # каждый день грузит на 7 дней вперед и получается накладка
    forecast_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    forecast_dataframe["month"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).month
    forecast_dataframe["hour"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).hour

    test_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(forecast_dataframe["datetime_msc"] < str(DATE_BEG))[0]
        ]
    )
    test_dataframe.drop(
        forecast_dataframe.index[
            np.where(forecast_dataframe["datetime_msc"] > str(DATE_END))[0]
        ],
        inplace=True,
    )

    # Строка ниже - из датафрейма для предикта оставляем
    # только гтп начинающиеся не с GK
    test_dataframe = test_dataframe[
        (test_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (test_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    test_dataframe = test_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )

    forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"]
                > str(datetime.datetime.today())
            )[0]
        ],
        inplace=True,
    )
    # Сортировка датафрейма по гтп и дате
    forecast_dataframe.sort_values(["gtp", "datetime_msc"], inplace=True)
    forecast_dataframe["datetime_msc"] = forecast_dataframe[
        "datetime_msc"
    ].astype("datetime64[ns]")
    logging.info(
        "forecast_dataframe и test_dataframe преобразованы в нужный вид"
    )

    #
    fact = fact_load(0, "DATEADD(HOUR, -365 * 24, DATEDIFF(d, 0, GETDATE()))")
    #
    # print(fact)

    forecast_dataframe = forecast_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )
    forecast_dataframe = forecast_dataframe.merge(
        fact,
        left_on=["gtp", "datetime_msc"],
        right_on=["gtp", "dt"],
        how="left",
    )
    # print(forecast_dataframe)
    # forecast_dataframe.to_excel('forecast_dataframe.xlsx')

    forecast_dataframe.dropna(subset=["fact"], inplace=True)
    forecast_dataframe.drop(["dt", "loadtime"], axis="columns", inplace=True)
    # print(forecast_dataframe)
    # print(test_dataframe)
    # forecast_dataframe.to_excel("forecast_dataframe.xlsx")
    # test_dataframe.to_excel("test_dataframe.xlsx")

    col_to_float = ["def_power"] + COL_PARAMETERS
    for col in col_to_float:
        forecast_dataframe[col] = forecast_dataframe[col].astype("float")
        test_dataframe[col] = test_dataframe[col].astype("float")

    col_to_int = ["month", "hour"]
    for col in col_to_int:
        forecast_dataframe[col] = forecast_dataframe[col].astype("int")

    logging.info("Датафреймы погоды и факта выработки склеены")
    return forecast_dataframe, test_dataframe


# Раздел подготовки прогноза на XGBoost
def prepare_forecast_xgboost(forecast_dataframe, test_dataframe):
    col_to_float = COL_PARAMETERS + ["def_power"]
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    # z["gtp"] = z["gtp"].str.replace("GKZV", "4")
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(["fact", "datetime_msc"], axis=1)
    # print(x)
    # x.to_excel("x.xlsx")
    y = z["fact"]

    predict_dataframe = test_dataframe.drop(
        ["datetime_msc", "loadtime"], axis=1
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    # predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
    #     "GKZV", "4"
    # )
    # predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )
    # print(predict_dataframe)
    # predict_dataframe.to_excel("predict_dataframe.xlsx")
    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")
    #
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.9
    )
    logging.info("Старт предикта на XGBoostRegressor")

    param = {
        "lambda": 0.10427064120338686,
        "alpha": 0.0023793948424012125,
        "colsample_bytree": 0.3,
        "subsample": 0.8,
        "learning_rate": 0.014,
        "n_estimators": 10000,
        "max_depth": 11,
        "random_state": 2020,
        "min_child_weight": 1,
    }
    reg = xgb.XGBRegressor(**param)
    regr = BaggingRegressor(base_estimator=reg, n_estimators=3, n_jobs=-1).fit(
        x_train, y_train
    )
    predict = regr.predict(predict_dataframe)
    test_dataframe["forecast"] = pd.DataFrame(predict)
    test_dataframe["forecast"] = test_dataframe["forecast"] * OVERVALUE_COEFF

    logging.info("Подготовлен прогноз на CatBoostRegressor")
    #
    # Обработка прогнозных значений
    # Обрезаем по максимум за месяц в часы
    max_month_dataframe = pd.DataFrame()
    date_cut = (
        datetime.datetime.today() + datetime.timedelta(days=-29)
    ).strftime("%Y-%m-%d")
    cut_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(forecast_dataframe["datetime_msc"] < str(date_cut))[0]
        ]
    )
    for gtp in test_dataframe.gtp.value_counts().index:
        max_month = (
            cut_dataframe.loc[
                cut_dataframe.gtp == gtp, ["fact", "hour", "gtp"]
            ]
            .groupby(by=["hour"])
            .max()
        )
        max_month.reset_index(inplace=True)
        max_month_dataframe = max_month_dataframe.append(
            max_month, ignore_index=True
        )
    # max_month_dataframe["hour"] = test_dataframe["hour"]
    test_dataframe = test_dataframe.merge(
        max_month_dataframe,
        left_on=["gtp", "hour"],
        right_on=["gtp", "hour"],
        how="left",
    )
    test_dataframe.fillna(0, inplace=True)
    test_dataframe["forecast"] = test_dataframe[
        ["forecast", "fact", "def_power"]
    ].min(axis=1)
    # Если прогноз отрицательный, то 0
    test_dataframe.forecast[test_dataframe.forecast < 0] = 0

    test_dataframe["forecast"] = np.where(
        test_dataframe["forecast"] == 0,
        (
            np.where(
                test_dataframe["fact"] > 0, np.NaN, test_dataframe.forecast
            )
        ),
        test_dataframe.forecast,
    )
    test_dataframe["forecast"].interpolate(
        method="linear", axis=0, inplace=True
    )
    test_dataframe["forecast"] = test_dataframe[["forecast", "fact"]].min(
        axis=1
    )

    test_dataframe.drop(
        ["fact", "month", "hour", "loadtime"], axis="columns", inplace=True
    )
    test_dataframe.drop(col_to_float, axis="columns", inplace=True)

    # Добавить к датафрейму столбцы с текущей датой и id прогноза
    # INSERT INTO treid_03.weather_foreca (gtp,dt,id_foreca,load_time,value)
    test_dataframe.insert(2, "id_foreca", "20")
    test_dataframe.insert(3, "load_time", datetime.datetime.now().isoformat())
    test_dataframe.rename(
        columns={"datetime_msc": "dt", "forecast": "value"},
        errors="raise",
        inplace=True,
    )
    # print(test_dataframe)

    # print(test_dataframe)
    # test_dataframe.to_excel(
    #     f'{pathlib.Path(__file__).parent.absolute()}/{(datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%d.%m.%Y")}_xgboost_2day_all.xlsx'
    # )
    logging.info(
        "Датафрейм с прогнозом выработки прошел обработку от нулевых значений"
        " и обрезку по макс за месяц"
    )
    #

    #
    # Запись прогноза в БД
    load_data_to_db("weather_foreca", 2, test_dataframe)
    #
    # Уведомление о подготовке прогноза
    telegram(0, "Open_Meteo: прогноз подготовлен")
    logging.info("Прогноз записан в БД treid_03")


def optuna_tune_params(forecast_dataframe, test_dataframe):
    # Подбор параметров через Optuna
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    # z["gtp"] = z["gtp"].str.replace("GKZV", "4")
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(["fact", "datetime_msc"], axis=1)
    # print(x)
    # x.to_excel("x.xlsx")
    y = z["fact"]

    predict_dataframe = test_dataframe.drop(
        ["datetime_msc", "loadtime"], axis=1
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    # predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
    #     "GKZV", "4"
    # )
    # predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )
    # print(predict_dataframe)
    # predict_dataframe.to_excel("predict_dataframe.xlsx")
    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    def objective(trial):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, train_size=0.8
        )
        # 'tree_method':'gpu_hist',
        # this parameter means using the GPU when training our model
        # to speedup the training process
        param = {
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
            ),
            "n_estimators": 10000,
            "max_depth": trial.suggest_categorical(
                "max_depth", [5, 7, 9, 11, 13, 15, 17]
            ),
            "random_state": trial.suggest_categorical(
                "random_state", [500, 1000, 1500, 2000]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        }

        reg = xgb.XGBRegressor(**param)
        reg.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=200,
        )
        prediction = reg.predict(predict_dataframe)
        score = reg.score(x_train, y_train)
        return score

    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=3600)
    optuna_vis = optuna.visualization.plot_param_importances(study)
    print(optuna_vis)
    print("Number of completed trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("\tBest Score: {}".format(trial.value))
    print("\tBest Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Сам процесс работы разбит для удобства по функциям
# чтобы если погода загрузилась, а прогноз не подготовился,
#  то чтобы не тратить лимит запросов и не засорять базу,
# закомменчиваем первую функцию и разбираемся дальше сколько угодно попыток.
# 1 - load_forecast_to_db - загрузка прогноза с сайта и запись в бд
# 2 - prepare_datasets_to_train - подготовка датасетов для обучения модели,
# переменным присваиваются возвращаемые 2 датафрейма и список столбцов,
# необходимо для работы следующих функций.
# 3 - optuna_tune_params - подбор параметров для модели через оптуну
# необходимо в нее передать 2 датафрейма из предыдущей функции.
# 4 - prepare_forecast_xgboost - подготовка прогноза,

# # 1
load_forecast_to_db(DATE_BEG, DATE_END, COL_PARAMETERS)
# # 2
forecast_dataframe, test_dataframe = prepare_datasets_to_train()
# # 3
# optuna_tune_params(forecast_dataframe, test_dataframe)
# # 4
prepare_forecast_xgboost(forecast_dataframe, test_dataframe)

print("Время выполнения:", datetime.datetime.now() - start_time)
