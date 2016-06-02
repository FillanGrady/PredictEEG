#import tensorflow as tf
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
    ONLY call next_seizure when time is higher than the last time you called it
    """
    def __init__(self, seizure_information):
        self.seizure_information = seizure_information
        self.file_number = -1
        self.next_seizure_start_time = -1
        self.next_seizure_end_time = -1
    
    def next_seizure(self, time):
        """
        Returns time (in seconds) until next seizure
        """
        if time < self.next_seizure_start_time:
            return self.next_seizure_start_time - time
        elif time < self.next_seizure_end_time:
            return 0
        else:
            self.file_number += 1
            try:
                while self.seizure_information[self.file_number].number_of_seizures == 0:
                    self.file_number += 1
            except IndexError as e:
                self.next_seizure_start_time = float("inf")
                self.next_seizure_end_time = float("inf")
                return self.next_seizure(time)
            self.next_seizure_start_time = self.seizure_information[self.file_number].seizure_start_time
            self.next_seizure_end_time = self.seizure_information[self.file_number].seizure_end_time
            return self.next_seizure(time)


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
        self.index = -1

    def next_eeg(self):
        """
        Returns the EEG readings and time for the next timepoint
        Use to iterate through data
        Returns a numpy array and a float
        """
        self.index += 1
        if self.index >= self.times.shape[0]:
            self.file_index += 1
            if self.file_index > len(self.seizure_information):
                raise StopIteration
            try:
                next_eeg_file_path = 'chb%02d_%02d.edf' % \
                                 (self.patient_number, self.seizure_information[self.file_index].patient_recording)
                raw = mne.io.read_raw_edf(next_eeg_file_path)
            except IOError as e:
                print e
                raise IOError("Download missing file")
            self.data, self.times = raw[:, :]
            self.times = np.array(self.times)
            self.data = self.data.T
            self.times += self.seizure_information[self.file_index].start_time
            self.index = -1
            return self.next_eeg()
        else:
            return self.data[self.index, :], self.times[self.index]


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


def get_seizure_information(patient_number, patient_recording):
    """
    Returns information about seizure file"""
    summary_file_path = 'chb%02d-summary.txt' % patient_number
    eeg_file_path = 'chb%02d_%02d.edf' % (patient_number, patient_recording)
    found_file = False
    buffer = []
    with open(summary_file_path) as summary_file:
        for line in summary_file:
            if not found_file:
                if eeg_file_path in line:
                    found_file = True
            else:
                if line != "\n":
                    buffer.append(line.strip())
                else:
                    return EEGInformation(patient_number, patient_recording, buffer)
    return None

if __name__ == '__main__':
    seizure_information = []
    number_files = 46
    for i in range(1, number_files+1):
        s = get_seizure_information(1, i)
        if s is not None:
            seizure_information.append(s)
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
    ns = NextSeizure(seizure_information)
    ne = NextEEG(1, seizure_information)
    try:
        while True:
            data, time = ne.next_eeg()
            time_to_next_seizure = ns.next_seizure(time)
    except StopIteration:
        print "Done!"