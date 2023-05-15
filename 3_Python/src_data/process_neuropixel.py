import os
import numpy as np
import pandas

# TODO: Funktionen nochmals prüfen
class NeuroPixelHandler:
    def __init__(self, path: str, filename, fs: int):
        self.path2data = path
        self.file_name = filename
        self.path2file = os.path.join(self.path2data, self.file_name)

        # --- Values for reading bin-file
        self.no_electrodes = 385
        # dX = dT* fs -1 --> dX = 199.999 (=5 s @30 kHz)
        self.dX = 19999
        self.no_bytes = 2
        self.file2read = None
        self.file2write = None

        self.sampling_rate = fs # 30e3

    def load_npy_file(self):
        """Loading data set from Neuropixel Recordings"""
        FileOrigin = self.path2file + '.npy'
        if os.path.exists(FileOrigin):
            print("Start pre-processing the results of KiloSort")

            # --- Loading the data
            clusters = np.load(self.path2file + "spike_clusters.npy")
            spike_times = np.load(self.path2file + "spike_times.npy") / self.sampling_rate
            spike_templates = np.load(self.path2file + "spike_templates.npy")
            ch_map = np.load(self.path2file + "channel_positions.npy")
            y_coords = np.load(self.path2file + "channel_positions.npy")
            data = pandas.read_csv(self.path2file + "cluster_groups.csv", sep='\t', low_memory=True)
            cids = []
            cfg = []
            for idx in range(len(data.cluster_id)):
                cids = np.append(cids, data.cluster_id[idx.numerator])
                cfg = np.append(cfg, data.group[idx.numerator])

            good_clusters = cids[cfg == 'good']
            good_indices = (np.in1d(clusters, good_clusters))

            real_spikes = spike_times[good_indices]
            real_clusters = clusters[good_indices]
            real_spike_templates = spike_templates[good_indices]

            counts_per_cluster = np.bincount(real_clusters)

            sort_idx = np.argsort(real_clusters)
            sorted_clusters = real_clusters[sort_idx]
            sorted_spikes = real_spikes[sort_idx]
            sorted_spike_templates = real_spike_templates[sort_idx]

            print("Data is readin")
        else:
            print("File does not exits")

    def load_bin_file_segment(self):
        """Processing bin-file: Time all, but specific electrode-wise """
        FileOrigin = self.path2file + '.bin'
        print("Read File:", FileOrigin)
        iteration = 0
        doRead = True
        OutFormat = 1
        loop_begin = 0
        loop_end = 10000

        while doRead:
            for idx in range(loop_begin, loop_end, 1):
                Path2Write = self.path2file + "_ELEC" + str(1 + idx.numerator)
                fileWR = open(Path2Write + ".bin", "wb")
                fileRD = open(FileOrigin, 'rb')

                iteration = 1
                doFile = True
                FileExists = False
                while doFile:
                    in1 = self.__read_file_row(fileRD)
                    if in1:
                        if (OutFormat == 1):
                            fileWR.write(in1[idx.numerator])
                        else:
                            if FileExists:
                                DataIn = np.load(Path2Write + ".npy")
                            else:
                                DataIn = np.zeros(0, dtype="int16")
                            DataIn = np.append(DataIn,
                                               int.from_bytes(in1[idx.numerator], 'little', signed=True))
                            np.save(Path2Write + ".npy", DataIn)
                            FileExists = True
                    else:
                        if (OutFormat == 1):
                            fileWR.close()
                        fileRD.close()
                        doFile = False

                # --- Zweite Abbruchbedingung
                iteration += 1
                print("... file " + str(iteration) + " of " + str(self.no_electrodes) + " done!")
                if iteration == self.no_electrodes:
                    doRead = False

    def load_bin_electrodewise(self):
        """Processing bin-file: Time all, but each electrode separated"""
        FileOrigin = self.path2file + '.bin'
        print("Read File:", FileOrigin)
        iteration = 0
        do_read = True
        OutFormat = 1
        loop_begin = 0
        loop_end = 10000

        file_exists = False
        fileRD = None
        while do_read:
            if iteration == 0:
                fileRD = open(FileOrigin, 'rb')
            else:
                data, state, col = self.__read_file_row(fileRD)

                if state == 1:
                    for idx in range(loop_begin, loop_end, 1):
                        Path2Write = self.path2file + "_ELEC" + str(1 + idx.numerator)
                        if OutFormat == 1:
                            if file_exists:
                                fileWR = open(Path2Write, 'ab')
                            else:
                                fileWR = open(Path2Write, 'wb')
                            for pos in range(self.dX):
                                fileWR.write(data[pos.numerator][idx.numerator])
                            fileWR.close()
                        # TODO: Save to npy einfügen
                    do_read = True
                    file_exists = True
                else:
                    do_read = False

            iteration += 1
        fileRD.close()

    def load_bin_timewindow(self):
        """Processing bin-file: Time window, all electrodes """
        FileOrigin = self.path2file + '.bin'
        print("Read File:", FileOrigin)
        iteration = 0
        do_read = True
        idx = 0
        fileRD = None

        while do_read:
            if iteration == 0:
                fileRD = open(FileOrigin, 'rb')
            else:
                data, state, col = self.__read_file_row(fileRD)

                if state == 1:
                    fileWR = open(self.path2file + str(1 + idx.numerator) + ".bin", 'ab')
                    for pos in range(self.dX):
                        fileWR.write(data[pos.numerator][idx.numerator])
                    fileWR.close()
                    do_read = True
                else:
                    do_read = False
            iteration += 1
        fileRD.close()

    def save_file(self):
        pass

    def __read_binfile(self, fileID):
        data = []
        col = 0
        state = 0

        do_read = True
        while do_read:
            in1 = self.__read_file_row(fileID)
            if in1:
                data.append(in1)
            else:
                do_read = False

            # --- Checking Abbruch-Bestimmung
            if col > self.dX:
                state = 1
                do_read = False
            elif not in1:
                state = 2
                do_read = False

            col += 1
            if col % 10000 == 0:
                print(col)

        # End processing of data
        return data, state, col

    def __read_file_row(self, fileID):
        in0 = fileID.read(self.no_bytes * self.no_electrodes)
        in1 = [in0[idx:idx + 2] for idx in range(0, len(in0), self.no_bytes)]

        return in1

if __name__ == "__main__":
    DataPathIn = "D:/0A_UCL_CortexLab_SingleProbe2017/Data"
    FileNameIn = "spike_times"
