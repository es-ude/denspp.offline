import os
import shutil
import glob

if __name__ == "__main__":
    data_type = "*NSP*.mat"
    path2data = "C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten"
    data_set = "10_Klaes_Caltech"

    folder_content = glob.glob(os.path.join(path2data, data_set, data_type))
    folder_content.sort()

    for name in folder_content:
        folder_name = str.split(name, '-NSP')
        path2final = os.path.join(path2data, data_set, folder_name[0])
        if not os.path.exists(path2final):
            os.mkdir(path2final)

        shutil.move(name, path2final)
