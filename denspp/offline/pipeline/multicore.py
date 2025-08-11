import numpy as np
from multiprocessing import Process, Queue
from os import cpu_count
from tqdm import tqdm
from logging import getLogger, Logger


def _core_task(rawdata: np.ndarray, id: int, func, result) -> None:
        result.put({id: func(rawdata)})


class MultiprocessingHandler:
    _logger: Logger
    __threads_core: list = list()
    _results: dict = dict()
    _num_cores: int = cpu_count()

    def __init__(self, num_workers: int) -> None:
        """Thread processor for analyzing data
        :param num_workers:    Integer with number of parallel workers
        :returns:              None
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self._max_num_workers = num_workers

    def __perform_single_threads(self, func, rawdata: np.ndarray | list, chnnl_id: list) -> None:
        self._logger.info('... processing data via single threading')

        self._results = dict()
        self.__threads_core = list()
        rslt = Queue()
        for idx, (chnnl, data)  in enumerate(tqdm(zip(chnnl_id, rawdata), ncols=100, desc='Progress Threads: ')):
            thread = Process(
                target=_core_task,
                kwargs=dict(
                    rawdata=data,
                    id=chnnl,
                    func=func,
                    result=rslt
            ))
            thread.start()
            thread.join()
            self._results.update(rslt.get())

    def __perform_multi_threads(self, func, data: np.ndarray | list, chnnl_id: list) -> None:
        num_iterations = int(np.ceil(len(chnnl_id) / self._max_num_workers))
        num_effective = num_iterations if num_iterations < self._num_cores else self._num_cores
        split_groups = [chnnl_id[i:i + num_effective] for i in range(0, len(chnnl_id), num_effective)]

        self._logger.info(f"... processing data with {self._max_num_workers} threading workers on {self._num_cores} cores")
        self._results = dict()
        for group in tqdm(split_groups, ncols=100, desc='Progress Threads: '):
            self.__threads_core = list()
            # --- Starting all threads
            rslt = Queue()
            for idx, group_num in enumerate(group):
                thread = Process(
                    target=_core_task,
                    kwargs=dict(
                        rawdata=data[idx],
                        id=group_num,
                        func=func,
                        result=rslt
                    )
                )
                self.__threads_core.append(thread)
                self.__threads_core[idx].start()

            # --- Waiting all threads are ready
            for thread in self.__threads_core:
                thread.join()
                self._results.update(rslt.get())

    def do_save_results(self, path2save: str) -> None:
        """Saving results in desired numpy format"""
        np.save(f'{path2save}/results.npy', self._results)

    def get_results(self) -> dict:
        """Return the signals after processing"""
        return self._results

    def do_processing(self, func, data: np.ndarray | list, chnnl_id: list) -> None:
        """Performing the data processing
        :param func:        Function to add Thread Processor for parallel processing
        :param data:        Numpy array with data to process (shape=[num. channels, num. samples])
        :param chnnl_id:    List wit hInteger with number of parallel workers
        """
        if self._max_num_workers > 1 and len(chnnl_id) > 1:
            self.__perform_multi_threads(func, data, chnnl_id)
        else:
            self.__perform_single_threads(func, data, chnnl_id)
