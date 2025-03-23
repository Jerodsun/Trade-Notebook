from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

dl(year="2024", pair="SPX/USD", time_frame="M1")
dl(year="2025", month="2", pair="SPXUSD", time_frame="M1")
