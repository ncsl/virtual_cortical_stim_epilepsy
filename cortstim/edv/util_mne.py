import mne


def create_mne_topostruct_from_numpy(datamat, chanlabels, samplerate, montage='standard_1020', **kwargs):
    # if montage not in mne.
    # create the info struct
    info = mne.create_info(ch_names=chanlabels.tolist(), ch_types=[
                                                                      'eeg'] * len(chanlabels), sfreq=samplerate)

    # create the raw object
    raw = mne.io.RawArray(data=datamat, info=info)

    # read in a standardized montage and set it
    montage = mne.channels.read_montage(montage)
    raw.set_montage(montage=montage)

    return raw
