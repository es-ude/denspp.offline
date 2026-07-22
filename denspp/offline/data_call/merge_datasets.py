from copy import copy
from datetime import datetime
from logging import Logger, getLogger
from os import makedirs
from pathlib import Path
from shutil import rmtree
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from denspp.offline import check_keylist_elements_any, get_path_to_project
from denspp.offline.data_call import CollectorH5, DataFromFile, LabelCollector, SettingsData

""" --------------------------------------------------------------------------------------------
INPUT DATA FILE STRUCTURE:

├── data_raw      # Raw signals
├── evnt_xpos     # Event positions in the raw signal
├── evnt_id       # Original labels / classes
├── fs_used       # Sampling rate
├── data_name     # Name of the dataset
└── label_exist   # Information, if labels exist


TEMPORARY DATA FILE STRUCTURE (temp_merge/):

temp_merge/
└── YYYY-MM-DD_Dataset-<data_name>.h5
    ├── waveform
    ├── xpos
    ├── label
    └── sampling_rate


FINAL DATA FILE STRUCTURE (dataset/):

dataset/
└── YYYY-MM-DD_Dataset-<data_name>_Merged.h5
    ├── data
    ├── class
    ├── position
    ├── sampling_rate
    ├── create_time
    ├── label_mapping_keys      # original labels
    └── label_mapping_values    # new labels
------------------------------------------------------------------------------------------------------------------"""


class MergeDataset:
    def __init__(
        self,
        pipeline: Any,
        dataloader: Any,
        settings_data: SettingsData,
        concatenate_id: bool = False,
    ) -> None:
        """Class for handling the merging process for generating datasets from transient input signals
        :param pipeline:                Construct of selected pipeline for processing data
        :param dataloader:              Construct of used Dataloader for getting and handling data
        :param settings_data:           Class SettingsData for configuring the data loader
        :param concatenate_id:          Do concatenation of the class number with increasing id number (useful for non-biological clusters)
        :return:                        None
        """
        self._logger: Logger = getLogger(__name__)

        self._dataloader = dataloader
        self._settings: SettingsData = settings_data
        self._pipeline = pipeline
        self._check_right_pipeline()

        self._path2save = get_path_to_project("temp_merge")
        self._do_label_concatenation = concatenate_id

    def _check_right_pipeline(self) -> None:
        if not check_keylist_elements_any(
            keylist=dir(self._pipeline),
            elements=["run_preprocessor", "run_classifier", "settings"],
        ):
            raise ImportError(
                "Wrong pipeline is implemented. It should include 'run_preprocessor' and 'run_classifier'."
            )

    def _generate_folder(self) -> None:
        if Path(self._path2save).exists():
            rmtree(self._path2save)
        makedirs(self._path2save, exist_ok=True)

    def _append_to_temp_dataset(
        self,
        h5f: h5py.File,
        name: str,
        data,
        dtype,
    ) -> None:
        """Appends one sample to a flat extendable temporary HDF5 dataset."""
        data = np.asarray(data, dtype=dtype)

        if name not in h5f:
            if data.ndim == 0:
                shape = (0,)
                maxshape = (None,)
                chunks = (8,)
            else:
                shape = (0, *data.shape)
                maxshape = (None, *data.shape)
                chunks = (8, *data.shape)

            h5f.create_dataset(
                name=name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype,
                compression="gzip",
                compression_opts=4,
            )

        dataset = h5f[name]
        old_len = dataset.shape[0]
        dataset.resize(old_len + 1, axis=0)
        dataset[old_len] = data

    def _save_frame_to_temp(self, frame, frame_index: int, h5f: h5py.File) -> None:
        """Saves frame content to the already opened flat temporary HDF5 file."""
        if "sampling_rate" not in h5f:
            h5f.create_dataset("sampling_rate", data=np.asarray(frame.sampling_rate))

        waveforms = np.asarray(frame.waveform, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :]

        positions = np.atleast_1d(np.asarray(frame.xpos, dtype=np.int32))
        labels = np.atleast_1d(np.asarray(frame.label, dtype=np.int32))

        if not (len(waveforms) == len(positions) == len(labels)):
            raise ValueError(
                f"Shape mismatch while saving temporary frame {frame_index}: \n"
                f"waveforms={len(waveforms)}, \n"
                f"positions={len(positions)}, \n"
                f"labels={len(labels)} \n"
            )

        for waveform, xpos, label in zip(waveforms, positions, labels):
            self._append_to_temp_dataset(h5f, "waveform", waveform, np.float32)
            self._append_to_temp_dataset(h5f, "xpos", int(xpos), np.int32)
            self._append_to_temp_dataset(h5f, "label", int(label), np.int32)

    def _iter_frames_from_temp(self, path2file: str):
        """Iterator for reading the samples of a temporary HDF5 file.
        Only one waveform is held in memory at a time.
        """
        with h5py.File(path2file, "r") as h5f:
            waveforms = h5f["waveform"]
            positions = h5f["xpos"]
            labels = h5f["label"]
            sampling_rate = float(h5f["sampling_rate"][()])

            for index in range(len(waveforms)):
                yield {
                    "waveform": np.asarray(waveforms[index]),
                    "xpos": np.asarray(positions[index]),
                    "label": np.asarray(labels[index]),
                    "sampling_rate": sampling_rate,
                }

    def get_frames_from_dataset(self, process_points: list | None = None, xpos_offset: int = 0) -> None:
        """Tool for loading datasets in order to generate one new dataset (Step 1)
        :param process_points:      Taking the datapoints of the selected data set to process
        :param xpos_offset:         Integer as position offset for shifting label position of an event (only apply if label exists)
        :return:                    None
        """
        if process_points is None:
            process_points = []

        self._generate_folder()
        current_index = 0
        create_time = datetime.now().strftime("%Y-%m-%d")

        while True:
            sets0 = copy.copy(self._settings)
            try:
                sets0.data_point = (
                    current_index if not len(process_points) else process_points[current_index]
                )
            except IndexError:
                break

            datahandler = None
            try:
                datahandler = self._dataloader(sets0)
                datahandler.do_call()
                datahandler.do_resample()
                datahandler.do_cut()

                data: DataFromFile = datahandler.get_data()

                temp_filename = f"{create_time}_Dataset-{data.data_name}.h5"
                temp_path = Path(self._path2save) / temp_filename

                with h5py.File(temp_path, "w") as h5f:
                    frame_counter = 0

                    if data.label_exist:
                        # labeled dataset ------------------------------------------------------------------------------
                        pipeline = self._pipeline(data.fs_used, False)
                        for rawdata, xposition, label in tqdm(
                            zip(data.data_raw, data.evnt_xpos, data.evnt_id),
                            ncols=100,
                            desc="Progress (labeled): ",
                        ):
                            xpos_scaler = pipeline.fs_ana / data.fs_used
                            xpos_updated = np.floor(xpos_scaler * xposition).astype("int")
                            result = pipeline.run_preprocessor(rawdata, xpos_updated, xpos_offset)
                            frame_new = result["frames"]
                            frame_new.label = label
                            self._save_frame_to_temp(frame_new, frame_counter, h5f)
                            frame_counter += 1
                            del frame_new
                    else:
                        # unlabeled dataset ----------------------------------------------------------------------------
                        pipeline = self._pipeline(data.fs_used, False)
                        for rawdata in tqdm(data.data_raw, ncols=100, desc="Progress (unlabeled): "):
                            result = pipeline.run_preprocessor(data=rawdata)
                            if "frames" not in result:
                                raise KeyError("Pipeline result missing 'frames'")
                            frame = result["frames"]
                            for attr in ("waveform", "xpos", "label", "sampling_rate"):
                                if not hasattr(frame, attr):
                                    raise AttributeError(f"Frame missing attribute {attr}")
                            frame_new = result["frames"]
                            self._save_frame_to_temp(frame_new, frame_counter, h5f)
                            frame_counter += 1
                            del frame_new

                self._logger.info(f"Saved temporary file: {temp_path.name} with {frame_counter} frames")

            except (StopIteration, FileNotFoundError) as e:
                self._logger.info(f"Data loading stopped: {e}")
                break

            except Exception:
                self._logger.exception(f"Failed loading data for {sets0.data_point}")
                raise

            finally:
                if datahandler is not None:
                    del datahandler

            current_index += 1

    def merge_data_from_all_iteration(self) -> str:
        """Merge all temporary HDF5 files into one final dataset.
        The sampling rate is extracted from the first frame and stored as a scalar.
        """
        folder_content = sorted(Path(self._path2save).glob("*.h5"))
        if not folder_content:
            raise FileNotFoundError(f"No *.h5 files found in {self._path2save}")

        file_name = Path(folder_content[-1]).name

        path2folder = get_path_to_project("dataset")
        Path(path2folder).mkdir(parents=True, exist_ok=True)

        path2file = Path(path2folder) / f"{Path(file_name).stem}_Merged.h5"

        label_collector = LabelCollector()
        data_elec_counter = 0
        sampling_rate = None

        with h5py.File(path2file, "w") as h5f:
            data_collector = CollectorH5(h5f, "data")
            class_collector = CollectorH5(h5f, "class")
            position_collector = CollectorH5(h5f, "position")

            data_collector.define_datatype(np.float32)
            class_collector.define_datatype(np.int32)
            position_collector.define_datatype(np.int32)

            for path in folder_content:
                for data_elec in self._iter_frames_from_temp(str(path)):
                    if sampling_rate is None:
                        sampling_rate = float(data_elec["sampling_rate"])

                    waveforms = np.asarray(data_elec["waveform"], dtype=np.float32)
                    if waveforms.ndim == 1:
                        waveforms = waveforms[np.newaxis, :]

                    positions = np.atleast_1d(np.asarray(data_elec["xpos"], dtype=np.int32))
                    labels = np.atleast_1d(np.asarray(data_elec["label"], dtype=np.int32))

                    if not (len(waveforms) == len(positions) == len(labels)):
                        raise ValueError(
                            f"Shape mismatch in {path}: \n"
                            f"waveforms={len(waveforms)}, \n"
                            f"positions={len(positions)}, \n"
                            f"labels={len(labels)} \n"
                        )

                    for waveform, position, label in zip(waveforms, positions, labels):
                        # emulate offset by combining keys
                        if self._do_label_concatenation:
                            label_key = int(label)
                        else:
                            label_key = f"{path.stem}_{int(label)}"

                        new_label = label_collector.add(label_key)

                        data_collector.add(np.asarray(waveform, dtype=np.float32))
                        position_collector.add(int(position))
                        class_collector.add(int(new_label))

                    data_elec_counter += 1

            h5f.create_dataset(
                "create_time",
                data=np.array(
                    datetime.now().strftime("%Y-%m-%d"),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )

            label_mapping = label_collector.get_all()
            h5f.create_dataset(
                "label_mapping_keys",
                data=np.array(
                    list(label_mapping.keys()),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )
            h5f.create_dataset(
                "label_mapping_values",
                data=np.array(
                    list(label_mapping.values()),
                    dtype=np.int32,
                ),
            )

            if sampling_rate is not None:
                h5f.create_dataset("sampling_rate", data=sampling_rate)
            else:
                self._logger.warning("No frames found – sampling_rate not set.")

        self._logger.info(f"Final merged dataset saved to: {Path(path2file).name}")
        return str(path2file)
