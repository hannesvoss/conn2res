# -*- coding: utf-8 -*-
"""
For testing conn2res.tasks functionality
"""

import numpy as np
# import pytest

import conn2res.tasks as tasks
from conn2res.tasks import ReservoirPyTask, Conn2ResTask


class TestReservoirPyTask():

    def test_fetch_data(self):
        task_names = tasks.get_task_list('reservoirpy')

        for task_name in task_names:

            task = ReservoirPyTask(name=task_name)
            x, y = task.fetch_data(n_trials=10)

            assert isinstance(x, np.ndarray), "incorrect type for x"
            assert isinstance(y, np.ndarray), "incorrect type for y"
            assert len(x) == len(y) == 10, "length x does not match length y"


class TestConn2ResTask():

    def test_fetch_data(self):
        task_names = tasks.get_task_list('conn2res')

        for task_name in task_names:

            task = Conn2ResTask(name=task_name)
            x, y = task.fetch_data(n_trials=10)

            assert isinstance(x, np.ndarray), "incorrect type for x"
            assert isinstance(y, np.ndarray), "incorrect type for y"
            assert len(x) == len(y) == 10, "length x does not match length y"
