import gspread
import numpy as np
import argparse
import copy
from datetime import timedelta, datetime
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt  # For plotting
# For saving figures to single pdf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
# *****************************************************************************
# Setting RC Parameters for figure size and fontsizes
import matplotlib.pylab as pylab
params = {'figure.figsize': (16, 12),
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'text.usetex': False,
          'lines.linewidth': 4,
          'font.family': 'serif',
          'font.serif': 'Georgia',
          'font.size': 20,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'axes.grid.axis': 'both',
          'axes.grid.which': 'both',
          'axes.grid': True,
          'grid.color': 'xkcd:cement',
          'grid.alpha': 0.3,
          'lines.markersize': 6,
          'legend.borderpad': 0.2,
          'legend.fancybox': True,
          'legend.fontsize': 'medium',
          'legend.framealpha': 0.8,
          'legend.handletextpad': 0.5,
          'legend.labelspacing': 0.33,
          'legend.loc': 'best',
          'savefig.dpi': 140,
          'savefig.bbox': 'tight',
          'pdf.compression': 9}
pylab.rcParams.update(params)

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name(
                                        'bodymeasurements-d291f81a84a8.json',
                                        scope)


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
        return np.nan
    else:
        return float(data[ii][ind])


def removeNan(tt, mWater, sWater):
    ii = 0
    ttWater = copy.deepcopy(tt)
    while(ii < len(mWater)):
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
    musPer = np.zeros(noEntries)
    for ii in range(noEntries):
        weight[ii] = parse(data, ii, 1)
        water[ii] = parse(data, ii, 2)
        fat[ii] = parse(data, ii, 3)
        muscle[ii] = parse(data, ii, 4)
        musPer[ii] = muscle[ii] * 100 / weight[ii]
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
    mMusPer = np.zeros(noWeeks)
    sWeight = np.zeros(noWeeks)
    sWater = np.zeros(noWeeks)
    sFat = np.zeros(noWeeks)
    sMuscle = np.zeros(noWeeks)
    sMusPer = np.zeros(noWeeks)
    nomStdWater = []
    nomStdFat = []
    nomStdMuscle = []
    nomStdMusPer = []
    for ii in range(len(weekList)-1):
        mWeight[ii] = np.nanmean(weight[weekList[ii]:weekList[ii+1]])
        mWater[ii] = np.nanmean(water[weekList[ii]:weekList[ii+1]])
        mFat[ii] = np.nanmean(fat[weekList[ii]:weekList[ii+1]])
        mMuscle[ii] = np.nanmean(muscle[weekList[ii]:weekList[ii+1]])
        mMusPer[ii] = np.nanmean(musPer[weekList[ii]:weekList[ii+1]])
        sWeight[ii] = np.nanstd(weight[weekList[ii]:weekList[ii+1]])
        if np.count_nonzero(~np.isnan(water[weekList[ii]:weekList[ii+1]])) > 3:
            nomStdWater += [np.nanstd(water[weekList[ii]:weekList[ii+1]])]
        if np.count_nonzero(~np.isnan(fat[weekList[ii]:weekList[ii+1]])) > 3:
            nomStdFat += [np.nanstd(fat[weekList[ii]:weekList[ii+1]])]
        if np.count_nonzero(~np.isnan(
                                    muscle[weekList[ii]:weekList[ii+1]])) > 3:
            nomStdMuscle += [np.nanstd(muscle[weekList[ii]:weekList[ii+1]])]
        if np.count_nonzero(~np.isnan(
                                    musPer[weekList[ii]:weekList[ii+1]])) > 3:
            nomStdMusPer += [np.nanstd(musPer[weekList[ii]:weekList[ii+1]])]

    nomStdWater = np.mean(np.array(nomStdWater))
    nomStdFat = np.mean(np.array(nomStdFat))
    nomStdMuscle = np.mean(np.array(nomStdMuscle))
    nomStdMusPer = np.mean(np.array(nomStdMusPer))
    for ii in range(len(weekList)-1):
        if np.count_nonzero(~np.isnan(water[weekList[ii]:weekList[ii+1]])) < 4:
            sWater[ii] = nomStdWater
        else:
            sWater[ii] = np.nanstd(water[weekList[ii]:weekList[ii+1]])
        if np.count_nonzero(~np.isnan(fat[weekList[ii]:weekList[ii+1]])) < 4:
            sFat[ii] = nomStdFat
        else:
            sFat[ii] = np.nanstd(fat[weekList[ii]:weekList[ii+1]])
        if np.count_nonzero(~np.isnan(
                                    muscle[weekList[ii]:weekList[ii+1]])) < 4:
            sMuscle[ii] = nomStdMuscle
        else:
            sMuscle[ii] = np.nanstd(muscle[weekList[ii]:weekList[ii+1]])
        if np.count_nonzero(~np.isnan(
                                    musPer[weekList[ii]:weekList[ii+1]])) < 4:
            sMusPer[ii] = nomStdMusPer
        else:
            sMusPer[ii] = np.nanstd(musPer[weekList[ii]:weekList[ii+1]])

    mWeight[-1] = np.nanmean(weight[weekList[-1]:])
    mWater[-1] = np.nanmean(water[weekList[-1]:])
    mFat[-1] = np.nanmean(fat[weekList[-1]:])
    mMuscle[-1] = np.nanmean(muscle[weekList[-1]:])
    mMusPer[-1] = np.nanmean(musPer[weekList[-1]:])
    sWeight[-1] = np.nanstd(weight[weekList[-1]:])
    sWater[-1] = np.nanmax([np.nanstd(water[weekList[-1]:]), 1])
    sFat[-1] = np.nanmax([np.nanstd(fat[weekList[-1]:]), 1])
    sMuscle[-1] = np.nanmax([np.nanstd(muscle[weekList[-1]:]), 1])
    sMusPer[-1] = np.nanmax([np.nanstd(musPer[weekList[-1]:]), 1])

    ttWater, mWater, sWater = removeNan(tt, mWater, sWater)
    ttFat, mFat, sFat = removeNan(tt, mFat, sFat)
    ttMuscle, mMuscle, sMuscle = removeNan(tt, mMuscle, sMuscle)
    ttMusPer, mMusPer, sMusPer = removeNan(tt, mMusPer, sMusPer)

    fig, lbs = plt.subplots(figsize=[16, 12])
    fig.subplots_adjust(right=0.75)
    perc = lbs.twinx()
    make_patch_spines_invisible(perc)
    perc.spines["right"].set_visible(True)

    we, = lbs.plot(tt, mWeight, "m-", lw=1.5, marker='*', label="Weight")
    fa, = perc.plot(ttFat, mFat, "y-", lw=1.5, marker='D', label="Fat %")
    wa, = perc.plot(ttWater, mWater, "b-", lw=1.5, marker='o', label="Water %")
    mp, = perc.plot(ttMusPer, mMusPer,  color='brown', lw=1.5, marker='v',
                    label="Muscle weight %")
    mu, = lbs.plot(ttMuscle, mMuscle, "r-", lw=1.5, marker='s',
                   label="Muscle Mass")

    lbs.fill_between(tt, mWeight - sWeight, mWeight + sWeight, color='m',
                     alpha=0.2)
    perc.fill_between(ttFat, mFat - sFat, mFat + sFat, color='y', alpha=0.2)
    perc.fill_between(ttWater, mWater - sWater, mWater + sWater, color='b',
                      alpha=0.2)
    perc.fill_between(ttMusPer, mMusPer - sMusPer, mMusPer + sMusPer,
                      color='brown', alpha=0.2)
    lbs.fill_between(ttMusPer, mMuscle - sMuscle, mMuscle + sMuscle, color='r',
                     alpha=0.2)

    lbs.grid(which='both')
    lbs.grid(axis='y', color='c')
    lbs.grid(axis='x', color='k')
    perc.grid(which='both')
    perc.grid(axis='y', color='orange')

    lbs.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))

    ybox1 = TextArea("Muscle",
                     textprops=dict(color="r", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox2 = TextArea(" and ",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox3 = TextArea("Weight",
                     textprops=dict(color="m", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox4 = TextArea(" [",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox5 = TextArea("lbs",
                     textprops=dict(color="c", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox6 = TextArea("]",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox = VPacker(children=[ybox6, ybox5, ybox4, ybox1, ybox2, ybox3],
                   align="bottom", pad=0, sep=5)
    lbs_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                 bbox_to_anchor=(-0.08, 0.3),
                                 bbox_transform=lbs.transAxes, borderpad=0.)
    lbs.add_artist(lbs_ybox)

    ybox1 = TextArea("Muscle",
                     textprops=dict(color="brown", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox2 = TextArea("and ",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox3 = TextArea("Water, ",
                     textprops=dict(color="b", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox4 = TextArea("Fat, ",
                     textprops=dict(color="y", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox5 = TextArea(" [",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox6 = TextArea("%",
                     textprops=dict(color="orange", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox7 = TextArea("]",
                     textprops=dict(color="k", size=24, rotation=90,
                                    ha='left', va='bottom'))
    ybox = VPacker(children=[ybox7, ybox6, ybox5, ybox1, ybox2, ybox3, ybox4],
                   align="bottom", pad=0, sep=5)
    perc_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                  bbox_to_anchor=(1.08, 0.3),
                                  bbox_transform=lbs.transAxes, borderpad=0.)
    perc.add_artist(perc_ybox)

    lbs.set_title('Body Measurements')

    fig.autofmt_xdate()

    figList = [fig]

    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(tt, mWeight, "m-", lw=1.5, marker='*', label="Weight")
    ax.fill_between(tt, mWeight - sWeight, mWeight + sWeight, color='m',
                    alpha=0.2)
    ax.set_ylabel('Weight [lbs]')
    ax.set_title('Body Weight')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
    fig.autofmt_xdate()
    figList += [fig]

    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(ttFat, mFat, "y-", lw=1.5, marker='*', label="Fat")
    ax.fill_between(ttFat, mFat - sFat, mFat + sFat, color='y',
                    alpha=0.2)
    ax.set_ylabel('Fat Percentage [%]')
    ax.set_title('Body Fat Percentage')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
    fig.autofmt_xdate()
    figList += [fig]

    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(ttWater, mWater, "b-", lw=1.5, marker='*', label="Water")
    ax.fill_between(ttWater, mWater - sWater, mWater + sWater, color='b',
                    alpha=0.2)
    ax.set_ylabel('Water Percentage [%]')
    ax.set_title('Body Water Percentage')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
    fig.autofmt_xdate()
    figList += [fig]

    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(ttMusPer, mMusPer, color='brown', lw=1.5, marker='*',
            label="MusPer")
    ax.fill_between(ttMusPer, mMusPer - sMusPer, mMusPer + sMusPer,
                    color='brown', alpha=0.2)
    ax.set_ylabel('Muscle Mass Percentage [%]')
    ax.set_title('Body Muscle Mass Percentage')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
    fig.autofmt_xdate()
    figList += [fig]

    fig, ax = plt.subplots(figsize=[16, 12])
    ax.plot(ttMuscle, mMuscle, "r-", lw=1.5, marker='*', label="Muscle")
    ax.fill_between(ttMuscle, mMuscle - sMuscle, mMuscle + sMuscle, color='r',
                    alpha=0.2)
    ax.set_ylabel('Muscle Mass [lbs]')
    ax.set_title('Body Muscle Mass')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
    fig.autofmt_xdate()
    figList += [fig]

    pp = PdfPages('./public/BodyMeasurementsTimeSeries.pdf')
    for ii, f in enumerate(figList):
        pp.savefig(f, bbox_inches='tight')
        f.savefig('./public/BodyMeasurementsTimeSeries_pg' + str(ii) + '.svg',
                  bbox_inches='tight')
    pp.close()


if __name__ == "__main__":
    bodyMeasurements(credentials)
