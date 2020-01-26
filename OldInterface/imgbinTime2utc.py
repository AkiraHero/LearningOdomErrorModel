import pandas as pd
from datetime import datetime
file = "/home/akira/poss-server/dataprocessing/campus0107/camera-mono/image_0.15/imgbinTime.txt"
outfile = "/home/akira/poss-server/dataprocessing/campus0107/camera-mono/utc_timestamp_usec.csv"
arr = pd.read_csv(file, header=None).values[:, 0]
hours = (arr * 1e3 / (3600 * 1e3)).astype(int)
minute = (arr * 1e3 % (3600 * 1e3) / (60 * 1e3)).astype(int)
secs = (arr * 1e3 % (3600 * 1e3) % (60 * 1e3) / 1e3).astype(int)
usecs = (arr * 1e3 % (3600 * 1e3) % (60 * 1e3) % 1e3 * 1e3).astype(int)
with open(outfile, 'w') as f:
    for i in zip(hours, minute, secs, usecs):
        t = datetime(2019, 1, 7, i[0], i[1], i[2], i[3])
        timestamp_sec = datetime.timestamp(t)
        timestamp_us = int(timestamp_sec * 1e6)
        s = str(timestamp_us)
        f.write(s + '\n')

