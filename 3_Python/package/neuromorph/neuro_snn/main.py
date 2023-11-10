import package.neuromorph.neuro_snn.scr.net_utility as nutil
import scr.plot_snn as show
import matplotlib.pyplot as plt

path2mat = "data/Martinez_2009/2023-04-17_Dataset01_SimDaten_Martinez2009_Sorted"
# variabeles for Dataloader
load_dataset = 3  # 1: mnist, 2: SpAIke, 3: time Variant SpAIke,
batch_size = 256
subset = 10  # reduce size of mnist by factor
valid_size = 0.1  # create validationset size out of dataset
exclude_list = []  # exclude Labels from SpAIke dataset
dt = 2e-6 # minimal time diverence between two spikes
sample_rate = 20e3 # sample rait of input data
hold_input = 3

# learning parameter
num_steps = 5  # time steps to hold input data
num_sample = 3

model_path = 'run/'
model_prefix = '20230625_144156' # ohne 0 frame
model_type = 'Kombinierter ausgang_8.0steps'


def main():
    net = nutil.load_net('model', '2023.02.28.16:27_cnn_SpAIke_plot_paper')

    #load dataset
    match load_dataset:  # 1: mnist, 2: SpAIke 3: SpAIke timebased, defold none
        case 1:
            print("load mnist dataset")
            net.load_mnist(batch_size=batch_size, subset=subset, valid_size=valid_size)
            net.set_time_invariant(num_steps=num_steps)
        case 2:
            print("load Spaike dataset")
            net.load_SpAIke(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
                            exclude_list=exclude_list)
            net.set_time_invariant(num_steps=num_steps)
        case 3:
            print("load time variant Spaike")
            #net.load_SpAIke_time(path=path2mat, batch_size=batch_size, subset=subset, valid_size=valid_size,
            #                     exclude_list=exclude_list, hold_data=hold_input, sample_rate=sample_rate,
            #                     dt=dt)  # hold_input
            #net.set_time_variant()
        case _:
            print("\033[91mNot implemented dataset\033[0m")
            return

    # show trainings loss
    print(net.dataset_name)
    show.loss_acc_plot(net=net)
    plt.show()
    return

    spk, targets, inputs = net.combine_input_and_gen_out(3)

    print(targets)
    show.plot_SpAIke_combined(spk[:, 0], inputs, targets, 256, path=net.get_path() + "figures/",
                              save_name=f'spk_combinaton_target_{net.label[targets[0]]}_{net.label[targets[1]]}_{net.label[targets[2]]}_with empty')
    plt.show()

if __name__ == '__main__':
    main()



