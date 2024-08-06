import os
import datetime as dt

import cdsapi


class WindByPressureLevels:

    """
    Home Page: https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
    """

    def __init__(self, pressure_level: int, destination_dir: str):
        self.client = cdsapi.Client()
        self.pressure_level: int = pressure_level
        self.destination_dir: str = destination_dir
        os.makedirs(name=f'{self.destination_dir}/{self.pressure_level}', exist_ok=True)

    def retrieve_by_date(
        self,
        year: int,
        month: int,
        day: int,
    ):
        year: str = str(year).zfill(2)
        month: str = str(month).zfill(2)
        day: str = str(day).zfill(2)
        filename: str = f'{year}{month}{day}.grib'
        
        request = {
            'product_type': ['reanalysis'],
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'year': [year],
            'month': [month],
            'day': [day],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', 
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
            ],
            'pressure_level': ['1000'],
            'data_format': 'grib',
            'download_format': 'unarchived'
        }

        self.client.retrieve(
            name="reanalysis-era5-pressure-levels", request=request
        ).download(f'{self.destination_dir}/{self.pressure_level}/{filename}')

    def retrieve_by_range(
        self,
        from_date: str,
        to_date: str,
    ):
        from_date: dt.datetime = dt.datetime.strptime(from_date, "%Y%m%d")
        to_date: dt.datetime = dt.datetime.strptime(to_date, "%Y%m%d")

        on_date: dt.datetime = from_date
        while on_date <= to_date:
            self.retrieve_by_date(
                year=str(on_date.year).zfill(4),
                month=str(on_date.month).zfill(2),
                day=str(on_date.day).zfill(2)
            )
            on_date += dt.timedelta(days=1)



if __name__ == '__main__':

    for pressure in [1000]:     # [1000, 700, 400, 200, 50]:
        self = WindByPressureLevels(pressure_level=pressure, destination_dir='data/2d/era5/wind')
        # self.retrieve_by_date(year=2024, month=7, day=29)
        self.retrieve_by_range(from_date='20240628', to_date='20240731')


