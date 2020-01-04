import gspread
import numpy as np
from datetime import timedelta, datetime
import time
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt  # For plotting
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
# *****************************************************************************
# Setting RC Parameters for figure size and fontsizes
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (20, 10),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name(
                                        'BodyMeasurements-2685fd4ece78.json',
                                        scope)
def parse(data, ii, ind):
    if data[ii][ind] == '':
        return np.NaN
    else:
        return float(data[ii][ind])

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

gc = gspread.authorize(credentials)
sh = gc.open("Daily Body Measurements (Responses)")
wks = sh.get_worksheet(0)
ts = wks.col_values(1)[1:]
noEntries = len(ts)
data = wks.get_all_values()[1:][:]
weight = np.zeros(noEntries)
water = np.zeros(noEntries)
fat = np.zeros(noEntries)
muscle = np.zeros(noEntries)
for ii in range(noEntries):
    weight[ii] = parse(data, ii, 1)
    water[ii] = parse(data, ii, 2)
    fat[ii] = parse(data, ii, 3)
    muscle[ii] = parse(data, ii, 4)
for ii in range(noEntries):
    ts[ii] = datetime.strptime(ts[ii], '%m/%d/%Y %H:%M:%S')
weekList = [0]
refDay = datetime.strptime('12/22/2019', '%m/%d/%Y')
lastWeekNo = (ts[0] - refDay).days//7
for ii in range(1, noEntries):
    if (ts[ii] - refDay).days//7 - lastWeekNo > 0:
        weekList += [ii]
        lastWeekNo = ii

firstWeek = datetime.strptime('12/25/2019', '%m/%d/%Y')
tt = [firstWeek + timedelta(days=7*ii) for ii in range(len(weekList))]
noWeeks = len(tt)
mWeight = np.zeros(noWeeks)
mWater = np.zeros(noWeeks)
mFat = np.zeros(noWeeks)
mMuscle = np.zeros(noWeeks)
sWeight = np.zeros(noWeeks)
sWater = np.zeros(noWeeks)
sFat = np.zeros(noWeeks)
sMuscle = np.zeros(noWeeks)
print(weekList)
for ii in range(len(weekList)-1):
    mWeight[ii] = np.nanmean(weight[weekList[ii]:weekList[ii+1]])
    mWater[ii] = np.nanmean(water[weekList[ii]:weekList[ii+1]])
    mFat[ii] = np.nanmean(fat[weekList[ii]:weekList[ii+1]])
    mMuscle[ii] = np.nanmean(muscle[weekList[ii]:weekList[ii+1]])
    sWeight[ii] = np.nanstd(weight[weekList[ii]:weekList[ii+1]])
    sWater[ii] = np.nanstd(water[weekList[ii]:weekList[ii+1]])
    sFat[ii] = np.nanstd(fat[weekList[ii]:weekList[ii+1]])
    sMuscle[ii] = np.nanstd(muscle[weekList[ii]:weekList[ii+1]])

mWeight[-1] = np.nanmean(weight[weekList[-1]:])
mWater[-1] = np.nanmean(water[weekList[-1]:])
mFat[-1] = np.nanmean(fat[weekList[-1]:])
mMuscle[-1] = np.nanmean(muscle[weekList[-1]:])
sWeight[-1] = np.nanstd(weight[weekList[-1]:])
sWater[-1] = np.nanstd(water[weekList[-1]:])
sFat[-1] = np.nanstd(fat[weekList[-1]:])
sMuscle[-1] = np.nanstd(muscle[weekList[-1]:])

fig, lbs = plt.subplots(figsize=[16,12])
fig.subplots_adjust(right=0.75)
perc = lbs.twinx()
make_patch_spines_invisible(perc)
perc.spines["right"].set_visible(True)

we, = lbs.plot(tt, mWeight, "m-", lw=1.5, marker='*', label="Weight")
fa, = perc.plot(tt, mFat, "y-", lw=1.5, marker='D', label="Fat %")
wa, = perc.plot(tt, mWater, "b-", lw=1.5, marker='o', label="Water %")
mu, = lbs.plot(tt, mMuscle, "r-", lw=1.5, marker='s', label="Muscle Mass")

lbs.fill_between(tt, mWeight - sWeight, mWeight + sWeight, color='m', alpha=0.3)
perc.fill_between(tt, mFat - sFat, mFat + sFat, color='y', alpha=0.3)
perc.fill_between(tt, mWater - sWater, mWater + sWater, color='b', alpha=0.3)
lbs.fill_between(tt, mMuscle - sMuscle, mMuscle + sMuscle, color='r', alpha=0.3)

lbs.grid(which='both')
perc.grid(which='both')

lbs.xaxis.set_ticks(tt)
lbs.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))

ybox1 = TextArea("Muscle",
                 textprops=dict(color="r", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox2 = TextArea(" and ",
                 textprops=dict(color="k", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox3 = TextArea("Weight",
                 textprops=dict(color="m", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox4 = TextArea(" [lbs]",
                 textprops=dict(color="k", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox = VPacker(children=[ybox4, ybox1, ybox2, ybox3],
               align="bottom", pad=0, sep=5)
lbs_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                  bbox_to_anchor=(-0.08, 0.4),
                                  bbox_transform=lbs.transAxes, borderpad=0.)
lbs.add_artist(lbs_ybox)

ybox1 = TextArea("Water",
                 textprops=dict(color="b", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox2 = TextArea(" and ",
                 textprops=dict(color="k", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox3 = TextArea("Fat",
                 textprops=dict(color="y", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox4 = TextArea(" [%]",
                 textprops=dict(color="k", size=15, rotation=90,
                                ha='left', va='bottom'))
ybox = VPacker(children=[ybox4, ybox1, ybox2, ybox3],
               align="bottom", pad=0, sep=5)
perc_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                  bbox_to_anchor=(1.08, 0.4),
                                  bbox_transform=lbs.transAxes, borderpad=0.)
perc.add_artist(perc_ybox)

lbs.set_title('Body Measurements', fontsize=20)

fig.autofmt_xdate()
fig.savefig('BodyMeasurementsTimeSeries.pdf', bbox_inches='tight')

time.sleep(300)
