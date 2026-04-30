# Source - https://stackoverflow.com/a/33885940
# Posted by MaxNoe
# Retrieved 2026-03-05, License - CC BY-SA 3.0

import numpy as np
import csv

"""
This is just to check if the npy file is read correctly.
Otherwise we access the data from the npy file which is good because it aligns with the mask coordina te
"""

data = np.load('gk_raw_depth_meter.npy')
with open('gk_raw_depth_meter.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)