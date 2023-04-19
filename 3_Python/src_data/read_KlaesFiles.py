from src_data.process_klaes_files import KlaesHandler

# ------------------------------ MAIN PROGRAMM
if __name__ == '__main__':
    path = 'D:'
    folder_name = '10_Klaes_Caltech'
    transfer_alldata = False

    print(f"\nProcessing the datasets of KlaesLab (US, Caltech) for MERCUR Sp:AI:ke")
    klaes_data = KlaesHandler(path, folder_name)
    if not transfer_alldata:
        sel_folder = [18]
        # sel_folder = range(19, klaes_data.noDataSets)
    else:
        sel_folder = range(0, klaes_data.noDataSets)

    for idx, folder in enumerate(sel_folder):
        data_save = klaes_data.search_data(folder)

        for file in range(0, klaes_data.noDataPoints):
            neural_data = klaes_data.process_rawdata(data_save, file)
            neural_label = klaes_data.process_label_nev(data_save, file)
            # neural_label1 = klaes_data.process_label_mat(data_save, file)
            neural_sorted = klaes_data.process_label_sorted(data_save, file)
            neural_behaviour = klaes_data.process_behaviour(data_save)

            klaes_data.save_results(
                name=data_save['Name'][file],
                rawdata=neural_data,
                label=neural_label,
                sorted=neural_sorted,
                behaviour=neural_behaviour
            )