import numpy as np
from torch import sum, abs, rand
from unittest import TestCase, main
from denspp.offline.dnn.ptq_help import quantize_model_fxp
from denspp.offline.dnn.pytorch_config_model import DefaultSettingsTrainMSE



class TestPTQ(TestCase):
    # --- Make models
    settings_test = DefaultSettingsTrainMSE
    settings_test.model_name = 'CompareDNN_Autoencoder_v1_Torch'
    model_test = settings_test.get_model()
    model_test.eval()
    model_qunt = quantize_model_fxp(model_test, 12, 10)
    model_qunt.eval()
    input = 2* (rand(size=(100,32)) - 0.5)

    def test_result_diff_feature(self):
        dout_test = self.model_test(self.input)[0]
        dout_qunt = self.model_qunt(self.input)[0]
        dmae = sum(abs(dout_test - dout_qunt)).detach().numpy() / len(self.input)
        np.testing.assert_allclose(dmae, 0.0028, atol=0.002)


    def test_result_diff_construction(self):
        dout_qunt_constructed = self.model_qunt(self.input)[1]
        mae_loss = sum(abs(self.input - dout_qunt_constructed)).detach().numpy() / len(self.input)
        np.testing.assert_allclose(mae_loss, 16.661808, atol=2.5)


if __name__ == '__main__':
    main()
