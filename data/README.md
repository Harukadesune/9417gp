# Dataset: nem_weather_halfhourly_2010_2026.csv

One row per half-hour from 2010-01-01 00:30 to 2026-11-01 00:00, i.e. every interval of the
days 1 January 2010 to 31 October 2026. This file is fixed: it is the only data you may use
to train, tune or engineer features.

| Column | Description | Unit |
| --- | --- | --- |
| timestamp | End of the half-hour interval, market time (AEST, UTC+10, no daylight saving) | |
| nsw_demand, qld_demand, vic_demand | Average operational demand in the interval | MW |
| nsw_price, qld_price, vic_price | Average spot price in the interval (can be negative) | $/MWh |
| syd_temp, mel_temp, bne_temp | Air temperature at 2 m in Sydney, Melbourne, Brisbane | °C |
| syd_rh, mel_rh, bne_rh | Relative humidity at 2 m | % |
| syd_cloud, mel_cloud, bne_cloud | Total cloud cover | % |
| syd_solar, mel_solar, bne_solar | Shortwave solar radiation | W/m² |

Notes:
* Since 1 October 2021 the market settles every 5 minutes; those values are averaged to
  30-minute intervals so that the whole series is consistent.
* Prices occasionally reach several thousand $/MWh and are often negative in recent years.
  These values are genuine, not errors.
* Weather values are hourly and are linearly interpolated to half-hours.

Sources:
* Market data: Australian Energy Market Operator (AEMO), Aggregated price and demand data,
  https://www.aemo.com.au. Used with acknowledgement of AEMO as the source.
* Weather data: Open-Meteo historical weather API (https://open-meteo.com), licensed under
  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Values were interpolated to
  half-hours.
