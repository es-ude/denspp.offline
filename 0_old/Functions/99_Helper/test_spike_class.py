from unittest import TestCase
import os, sys

#Helping function
import pandas as pd

from load_mat_files import *

#class to test
from spike_class import *

class Testspike_data(TestCase):
    def setUp(self) -> None:
        #data = load_01_SimDaten_Martinez2009([1])[0]
        data = load_02_SimDaten_Pedreira2012([1])[0]
        self.spike_data = spike_dataclass(data)

    def test_get_spikeframe(self):
        indices_cl_0 = np.where(self.spike_data.cluster == 0)[0].tolist()[0:100]
        indices_cl_1 = np.where(self.spike_data.cluster == 1)[0].tolist()[0:100]
        indices_cl_2 = np.where(self.spike_data.cluster == 2)[0].tolist()[0:100]
        indices_cl_0 = indices_cl_0 + indices_cl_1 + indices_cl_2
        data = list()
        for i in indices_cl_0:
            alligned_frame = self.spike_data.get_spikeframe(self.spike_data.times[i])
            data.append(alligned_frame)
            #sns.lineplot(data=alligned_frame)
            #ind = [x for x in range(len(self.spike_data.times)) if self.spike_data.times[x] == self.spike_data.times[i]]
            #plt.title('Time: ' + str(self.spike_data.times[i]) + ',Cluster: ' + str(self.spike_data.cluster[ind][0]))
            #plt.show()
        sns.lineplot(data=data)
        plt.show()


    def test_alignment(self):
        pass

''' 
        sns.lineplot(data=alligned_frame)
        ind = [x for x in range(len(self.times)) if self.times[x] == spike_time]
        plt.title('Time: ' + str(spike_time) + ',Cluster: ' + str(self.cluster[ind][0]))
        plt.show()     
'''