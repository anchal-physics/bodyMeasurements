import gspread
import numpy as np
import argparse
import copy
from datetime import timedelta, datetime
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
'''
credentials = ServiceAccountCredentials.from_json_keyfile_name(
                                        'BodyMeasurements-2685fd4ece78.json',
                                        scope)'''


def grabInputArgs():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--type')
    parser.add_argument('--project_id')
    parser.add_argument('--private_key_id')
    parser.add_argument('--private_key')
    parser.add_argument('--client_email')
    parser.add_argument('--client_id')
    parser.add_argument('--auth_uri')
    parser.add_argument('--token_uri')
    parser.add_argument('--auth_provider_x509_cert_url')
    parser.add_argument('--client_x509_cert_url')
    return parser.parse_args()

def parse(data, ii, ind):
    if data[ii][ind] == '':
        return np.NaN
    else:
        return float(data[ii][ind])

def removeNan(tt, mWater, sWater):
    ii=0
    ttWater = copy.deepcopy(tt)
    while(ii<len(mWater)):
        if np.isnan(mWater[ii]):
            mWater = np.delete(mWater, ii)
            sWater = np.delete(sWater, ii)
            del ttWater[ii]
        else:
            ii = ii+1
    return ttWater, mWater, sWater

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def bodyMeasurements(credentials):
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
    firstWeek = datetime.strptime('12/25/2019', '%m/%d/%Y')
    tt = [firstWeek + timedelta(days=7*lastWeekNo)]
    for ii in range(1, noEntries):
        if (ts[ii] - refDay).days//7 - lastWeekNo > 0:
            weekList += [ii]
            lastWeekNo = (ts[ii] - refDay).days//7
            tt += [firstWeek + timedelta(days=7*lastWeekNo)]
    noWeeks = len(tt)
    mWeight = np.zeros(noWeeks)
    mWater = np.zeros(noWeeks)
    mFat = np.zeros(noWeeks)
    mMuscle = np.zeros(noWeeks)
    sWeight = np.zeros(noWeeks)
    sWater = np.zeros(noWeeks)
    sFat = np.zeros(noWeeks)
    sMuscle = np.zeros(noWeeks)
    for ii in range(len(weekList)-1):
        mWeight[ii] = np.nanmean(weight[weekList[ii]:weekList[ii+1]])
        mWater[ii] = np.nanmean(water[weekList[ii]:weekList[ii+1]])
        mFat[ii] = np.nanmean(fat[weekList[ii]:weekList[ii+1]])
        mMuscle[ii] = np.nanmean(muscle[weekList[ii]:weekList[ii+1]])
        sWeight[ii] = np.nanstd(weight[weekList[ii]:weekList[ii+1]])
        sWater[ii] = np.maximum(np.nanstd(water[weekList[ii]:weekList[ii+1]]), 1)
        sFat[ii] = np.maximum(np.nanstd(fat[weekList[ii]:weekList[ii+1]]), 1)
        sMuscle[ii] = np.maximum(np.nanstd(muscle[weekList[ii]:weekList[ii+1]]), 1)

    mWeight[-1] = np.nanmean(weight[weekList[-1]:])
    mWater[-1] = np.nanmean(water[weekList[-1]:])
    mFat[-1] = np.nanmean(fat[weekList[-1]:])
    mMuscle[-1] = np.nanmean(muscle[weekList[-1]:])
    sWeight[-1] = np.nanstd(weight[weekList[-1]:])
    sWater[-1] = np.maximum(np.nanstd(water[weekList[-1]:]), 1)
    sFat[-1] = np.maximum(np.nanstd(fat[weekList[-1]:]), 1)
    sMuscle[-1] = np.maximum(np.nanstd(muscle[weekList[-1]:]), 1)

    ttWater, mWater, sWater = removeNan(tt, mWater, sWater)
    ttFat, mFat, sFat = removeNan(tt, mFat, sFat)
    ttMuscle, mMuscle, sMuscle = removeNan(tt, mMuscle, sMuscle)

    fig, lbs = plt.subplots(figsize=[16,12])
    fig.subplots_adjust(right=0.75)
    perc = lbs.twinx()
    make_patch_spines_invisible(perc)
    perc.spines["right"].set_visible(True)

    we, = lbs.plot(tt, mWeight, "m-", lw=1.5, marker='*', label="Weight")
    fa, = perc.plot(ttFat, mFat, "y-", lw=1.5, marker='D', label="Fat %")
    wa, = perc.plot(ttWater, mWater, "b-", lw=1.5, marker='o', label="Water %")
    mu, = lbs.plot(ttMuscle, mMuscle, "r-", lw=1.5, marker='s', label="Muscle Mass")

    lbs.fill_between(tt, mWeight - sWeight, mWeight + sWeight, color='m', alpha=0.2)
    perc.fill_between(ttFat, mFat - sFat, mFat + sFat, color='y', alpha=0.2)
    perc.fill_between(ttWater, mWater - sWater, mWater + sWater, color='b', alpha=0.2)
    lbs.fill_between(ttMuscle, mMuscle - sMuscle, mMuscle + sMuscle, color='r', alpha=0.2)

    lbs.grid(which='both')
    lbs.grid(axis='y', color='c')
    perc.grid(which='both')
    perc.grid(axis='y', color='orange')

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
    ybox4 = TextArea(" [",
                     textprops=dict(color="k", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox5 = TextArea("lbs",
                     textprops=dict(color="c", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox6 = TextArea("]",
                     textprops=dict(color="k", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox = VPacker(children=[ybox6, ybox5, ybox4, ybox1, ybox2, ybox3],
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
    ybox4 = TextArea(" [",
                     textprops=dict(color="k", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox5 = TextArea("%",
                     textprops=dict(color="orange", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox6 = TextArea("]",
                     textprops=dict(color="k", size=15, rotation=90,
                                    ha='left', va='bottom'))
    ybox = VPacker(children=[ybox6, ybox5, ybox4, ybox1, ybox2, ybox3],
                   align="bottom", pad=0, sep=5)
    perc_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                      bbox_to_anchor=(1.08, 0.4),
                                      bbox_transform=lbs.transAxes, borderpad=0.)
    perc.add_artist(perc_ybox)

    lbs.set_title('Body Measurements', fontsize=20)

    fig.autofmt_xdate()
    fig.savefig('BodyMeasurementsTimeSeries.pdf', bbox_inches='tight')

if __name__ == "__main__": #triger input parser on call
    args = grabInputArgs()
    private_key = args.private_key.replace('SPACE', ' ')
    private_key = args.private_key.replace('NEWLINE', '\n')
    print(private_key)
    keyfile_dict = {
        "type": args.type,
        "project_id": args.project_id,
        "private_key_id": args.private_key_id,
        "private_key": private_key,
        "client_email": args.client_email,
        "client_id": args.client_id,
        "auth_uri": args.auth_uri,
        "token_uri": args.token_uri,
        "auth_provider_x509_cert_url": args.auth_provider_x509_cert_url,
        "client_x509_cert_url": args.client_x509_cert_url
    }
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                                                                   keyfile_dict,
                                                                   scopes=scope)
    bodyMeasurements(credentials)
