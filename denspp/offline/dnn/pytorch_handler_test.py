import numpy as np
from unittest import TestCase, main
from denspp.offline.dnn.pytorch_handler import logic_combination


# --- Info: Function have to start with test_*
class TestPytrochHandler(TestCase):
    def test_logic_combination(self):
        true = np.array((0, 1, 1, 2, 0, 1, 1, 2, 1, 1, 2, 3, 3, 0, 1, 0, 2, 3), dtype=np.uint8)
        pred = np.array((0, 1, 2, 2, 0, 1, 1, 2, 1, 1, 2, 2, 3, 0, 1, 0, 2, 3), dtype=np.uint8)
        tran = [[0, 2], [1, 3]]

        rslt_true, rslt_pred = logic_combination(
            true_labels=true,
            pred_labels=pred,
            translate_list=tran
        )
        chck_true = np.array((0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1), dtype=np.uint8)
        chck_pred = np.array((0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1), dtype=np.uint8)
        self.assertTrue(np.array_equal(chck_true, rslt_true) and np.array_equal(chck_pred, rslt_pred))


if __name__ == '__main__':
    main()
