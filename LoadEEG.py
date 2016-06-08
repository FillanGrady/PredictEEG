import mne
import numpy as np

"""
Terminology:
One recording is an EEG reading session, usually about 24 hours in length
One file is one chunk of a recording; usually 1 hour in length
There is a small break between the end of one file and the beginning of the next, but these are negligible
"""


class NextSeizure:
    """
    This is a class for determining the amount of time until the next seizure
    It is only intended to be iterated through once, forward.
    ONLY call __getitem__ when time is higher than the last time you called it
    """
    def __init__(self, seizure_information):
        self.seizure_information = seizure_information
        self.file_number = -1
        self.next_seizure_start_time = -1
        self.next_seizure_end_time = -1

    def load_next_seizure(self):
        self.file_number += 1
        try:
            while self.seizure_information[self.file_number].number_of_seizures == 0:
                self.file_number += 1
        except IndexError as e:
            self.next_seizure_start_time = float("inf")
            self.next_seizure_end_time = float("inf")
        self.next_seizure_start_time = self.seizure_information[self.file_number].seizure_start_time
        self.next_seizure_end_time = self.seizure_information[self.file_number].seizure_end_time

    def time_until_next_seizure(self, time):
        """
        :time: integer time
        :return: time until next seizure
        """
        if time < self.next_seizure_start_time:
            return self.next_seizure_start_time - time
        if time < self.next_seizure_end_time:
            return 0
        else:
            self.load_next_seizure()
            return self.time_until_next_seizure(time)

    def time_until_next_seizure_array(self, time_array, number_output_neurons):
        """
        :time_array: numpy array of times
        :return: numpy array of time until the next seizure
        """
        shape = list(time_array.shape)
        shape.append(number_output_neurons)
        time_until = np.zeros(shape)
        for indicies, value in np.ndenumerate(time_array):
            indicies = list(indicies)
            time_until_next_seizure = int(self.time_until_next_seizure(value) / 60) # In minutes
            if time_until_next_seizure < number_output_neurons - 1:
                indicies.append(int(self.time_until_next_seizure(value)))
                time_until[tuple(indicies)] = 1
            else:
                indicies.append(number_output_neurons - 1)
                time_until[tuple(indicies)] = 1
        return time_until


class NextEEG:
    """
    This class is also designed to be called iteratively
    If you keep on calling next_eeg, it'll iterate through the edf EEG files,
    and return the EEG readings and the time since the start of the recording
    """
    def __init__(self, patient_number, seizure_information):
        self.patient_number = patient_number
        self.seizure_information = seizure_information
        self.file_index = -1
        self.data = np.zeros((0, 0))
        self.times = np.zeros((0, 0))
        self.index = 0
        self.sampling_frequency = -1  # Sample per second
        self.load_new_file()

    def next_eeg_samples(self, number_samples):
        """
        Returns the EEG readings and time for the next timepoint
        Use to iterate through data
        Returns a numpy array and a float
        """
        if self.index + number_samples >= self.times.shape[0]:
            """
            Gone over the end of the file.  Fill in the missing information with 0s, and switch to a new file
            """
            before_file_end = self.data[self.index:, :]
            after_file_end = np.zeros((self.index + number_samples - self.times.shape[0], self.data.shape[1]))
            eeg_data = np.concatenate([before_file_end, after_file_end])
            before_file_end = self.times[self.index:]
            after_file_end = np.zeros(self.index + number_samples)
            time_data = np.concatenate([before_file_end, after_file_end])

            self.load_new_file()
        else:
            eeg_data = self.data[self.index:self.index + number_samples, :]
            time_data = self.times[self.index:self.index + number_samples]
            self.index += number_samples
        return eeg_data, time_data

    def next_eeg_seconds(self, number_seconds):
        return self.next_eeg_samples(int(number_seconds * self.sampling_frequency))

    def load_new_file(self):
        self.file_index += 1
        if self.file_index > len(self.seizure_information):
            raise StopIteration
        try:
            next_eeg_file_path = 'chb%02d_%02d.edf' % \
                                 (self.patient_number, self.seizure_information[self.file_index].patient_recording)
            raw = mne.io.read_raw_edf(next_eeg_file_path)
        except IOError as e:
            print(e)
            raise IOError("Download missing file")
        self.data, self.times = raw[:, :]
        self.times = np.array(self.times)
        self.data = self.data.T
        self.sampling_frequency = (self.times.shape[0] - 1) / (self.times[-1] - self.times[0])
        self.times += self.seizure_information[self.file_index].start_time
        self.index = 0


class EEGInformation:
    """
    Used to store information about one EEG file
    """
    start_time = None
    end_time = None
    seizure_start_time = None
    seizure_end_time = None
    number_of_seizures = None
    patient_number = None
    patient_recording = None
    total_seconds = None

    def __init__(self, patient_number, patient_recording, buffer):
        self.patient_number = patient_number
        self.patient_recording = patient_recording
        for line in buffer:
            if "File Start Time" in line:
                time = [int(x) for x in line[line.index(": ") + 2:].split(':')]
                self.start_time = (time[0] * 60 * 60) + (time[1] * 60) + time[2]
            if "File End Time" in line:
                time = [int(x) for x in line[line.index(": ") + 2:].split(':')]
                self.end_time = (time[0] * 60 * 60) + (time[1] * 60) + time[2]
                if self.end_time < self.start_time:  # Overflowed, new day
                    self.end_time += 24 * 60 * 60  # Number of seconds in a day
                self.total_seconds = self.end_time - self.start_time
            if "Number of Seizures in File" in line:
                self.number_of_seizures = int("".join([x for x in line if x.isdigit()]))
                assert self.number_of_seizures <= 1  # Assume that there's only one seizure per file
                if self.number_of_seizures == 0:
                    self.seizure_start_time = None
                    self.seizure_end_time = None
                    return
            if "Seizure Start Time" in line:
                self.seizure_start_time = int("".join([x for x in line if x.isdigit()]))
            if "Seizure End Time" in line:
                self.seizure_end_time = int("".join([x for x in line if x.isdigit()]))

    def __str__(self):
        return "Time: %s --- %s\nNumber of Seizures: %s\nSeizure Time: %s-%s" %\
               (self.start_time, self.end_time, self.number_of_seizures, self.seizure_start_time, self.seizure_end_time)


def get_seizure_information(patient_number):
    """
    Returns information about eeg files in summary file
    """
    import fnmatch
    summary_file_path = 'chb%02d-summary.txt' % patient_number
    eeg_file_path = '*chb%02d_*.edf*' % patient_number
    found_file = False
    recording_number = -1
    buffer = []
    seizure_information = []
    with open(summary_file_path) as summary_file:
        for line in summary_file:
            if not found_file:
                if fnmatch.fnmatch(line, eeg_file_path):
                    found_file = True
                    recording_number = int(line[line.index("chb%02d_" % patient_number) + 6:line.index(".edf")])
                    buffer = []
            else:
                if line != "\n":
                    buffer.append(line.strip())
                else:
                    seizure_information.append(EEGInformation(patient_number, recording_number, buffer))
                    found_file = False
                    recording_number = -1

    print("%s eeg files for patient %s" % (len(seizure_information), patient_number))

    #  Rearranges the times so that they start at time = 0, and to properly account for days passing
    start_recording = seizure_information[0].start_time
    for s in seizure_information:
        s.start_time -= start_recording
        s.end_time -= start_recording
    for i in range(len(seizure_information) - 1):
        while seizure_information[i].start_time > seizure_information[i + 1].start_time:
            seizure_information[i + 1].start_time += 24 * 60 * 60
            seizure_information[i + 1].end_time += 24 * 60 * 60
    for s in seizure_information:
        if s.seizure_start_time is not None:
            s.seizure_start_time += s.start_time
        if s.seizure_end_time is not None:
            s.seizure_end_time += s.start_time

    return seizure_information


def setup(patient_number):
    seizure_information = get_seizure_information(patient_number)
    ne = NextEEG(patient_number, seizure_information)
    ns = NextSeizure(seizure_information)
    return ne, ns

if __name__ == '__main__':
    ne, ns = setup(1)
    eeg_data, time_data = ne.next_eeg_seconds(5)
    time_until = ns.time_until_next_seizure_array(time_data, 31)
