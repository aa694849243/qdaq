# Loudness
import numpy as np
from mosqito.functions.loudness_zwicker.comp_loudness import comp_loudness
# Sharpness
from mosqito.functions.sharpness.comp_sharpness import comp_sharpness
# Roughness
from mosqito.functions.roughness_danielweber.comp_roughness import comp_roughness
from mosqito.functions.tonality_tnr_pr.comp_pr import comp_pr
from mosqito.functions.tonality_tnr_pr.comp_tnr import comp_tnr


class MosqitoCalc:
    def __init__(self, raw_data, input_fs):
        self.raw_data = raw_data
        self.input_fs = input_fs
        self.resampledata = self.resample_by_interpolation()

    def oned_resp(self, name, value, unit):
        res = {'name': name, 'value': value, 'unit': unit, 'indicatorDiagnostic': -1}
        return res

    def twod_resp(self, x_name, x_unit, x_value, y_name, y_unit, y_value):
        res = {'xNname': x_name,
               'xVvalue': list(x_value),
               'xUnit': x_unit,
               'yNname': y_name,
               'yVvalue': list(y_value),
               'yUnit': y_unit,
               'indicatorDiagnostic': -1}
        return res

    def threed_resp(self, x_name, x_unit, x_value, y_name, y_unit, y_value, z_name, z_unit, z_value):
        res = {'xNname': x_name,
               'xVvalue': x_value,
               'xUnit': x_unit,
               'yNname': y_name,
               'yVvalue': y_value,
               'yUnit': y_unit,
               'zNname': z_name,
               'zUnit': z_unit,
               'zVvalue': z_value,
               'indicatorDiagnostic': -1}
        return res

    def resample_by_interpolation(self):
        if self.input_fs != 48000:
            scale = 48000 / self.input_fs
            # calculate new length of sample
            n = round(len(self.raw_data) * scale)
            # use linear interpolation
            # endpoint keyword means than linspace doesn't go all the way to 1.0
            # If it did, there are some off-by-one errors
            # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
            # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
            # Both are OK, but since resampling will often involve
            # exact ratios (i.e. for 44100 to 22050 or vice versa)
            # using endpoint=False gets less noise in the resampled sound
            resampled_signal = np.interp(
                np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
                np.linspace(0.0, 1.0, len(self.raw_data), endpoint=False),  # known pos
                self.raw_data,  # known data points
            )
            return resampled_signal
        else:
            return self.raw_data[:]

    # Loudness calculation
    # Time varying signal
    def loudness_time_varying_signal(self):
        signal, fs = self.resampledata, 48000
        loudness_data = comp_loudness(False, signal, fs, field_type='free')
        n = loudness_data['values']
        time = np.linspace(0, 0.002 * (n.size - 1), n.size)
        return self.twod_resp('time', 's', time, 'loudness', 'sone', n)

    # Loudness calculation
    # Steady signal
    def loudness_steady_signal(self):
        signal, fs = self.resampledata, 48000
        loudness = comp_loudness(True, signal, fs, field_type='free')
        n = loudness['specific values']
        bark_scale = np.linspace(0, 24, num=len(n))
        return self.twod_resp('frequency', 'bark', bark_scale, 'loudness', 'sone', n)

    # Sharpness calculation
    # Time varying signal
    def sharpness_time_varying_signal(self):
        signal, fs = self.resampledata, 48000
        sharpness = comp_sharpness(False, signal, fs, method="din", skip=0.2)
        s = sharpness['values']
        time = np.linspace(0, 0.002 * (s.size - 1), s.size)
        return self.twod_resp('time', 's', time, 'sharpness', 'acum', s)

    # Sharpness calculation
    # Steady signal
    def sharpness_steady_signal(self):
        signal, fs = self.resampledata, 48000
        sharpness = comp_sharpness(True, signal, fs, method="din")
        return self.oned_resp('sharpness', sharpness['values'], 'acum')

    # Roughness calculation
    def roughness_calc(self):
        signal, fs = self.resampledata, 48000
        roughness = comp_roughness(signal, fs, overlap=0)
        return self.twod_resp('time', 's', roughness['time'], 'roughnes', 'asper', roughness['values'])

    # PR calculation
    def pr_time_varing_signal(self, freq):
        output = comp_pr(False, self.raw_data, freq, prominence=True)
        return self.threed_resp('time', 's', output['time'], 'Frequency', 'Hz', output['freqs'], 'TNR value', 'dB',
                                output['values'])

    # Tone-to-noise Ratio calculation
    def tnr_steady_signal(self, freq):
        output = comp_tnr(True, self.raw_data, freq, prominence=True)
        return self.twod_resp('Frequency', 'Hz', output['freqs'], 'TNR value', 'dB', output['values']),output['global value']


def judgesensor(task_info, sensor_index):
    sensorchannel = []
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.lower()[:3] in ['mic', 'umi', 'vib']:
            sensorchannel.append(channel_name)
    if sensorchannel and sensorchannel[sensor_index].lower()[:3] == 'mic':
        return True
    return False
