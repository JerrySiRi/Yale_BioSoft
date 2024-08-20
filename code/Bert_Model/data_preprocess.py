import os
import json
import pandas




def convert_files_to_txt(folder_path, output_file):

    files = sorted(os.listdir(folder_path))
    txt_files = [f for f in files if f.endswith('.txt')]

    txt_train = [txt_files[index] for index in range(0,int(0.9*len(txt_files)))]
    txt_dev = [txt_files[index] for index in range(int(0.9*len(txt_files))+1, len(txt_files))]

    # ann_files = [f for f in files if f.endswith('.ann')]

    taged_train_name = output_file + "/BIO_train.txt"
    taged_test_name = output_file + "/BIO_test.txt"
    taged_dev_name = output_file +"/BIO_dev.txt"


    with open(taged_dev_name, "w", encoding="utf-8") as taged: # Use writelines to write list
        
        txt_files = txt_dev

        # Process .txt & .ann files
        for txt in txt_files:

            BIO_taged = []
            txt_with_index_content = []
            ann_label_list = [] # (index, name)
            ann_label_dict = dict()

            basename = os.path.splitext(txt)[0] # split filtname and extract its name without extension
            ann = basename + ".ann"
            with open(os.path.join(folder_path, txt), 'r', encoding='utf-8') as file_txt:
                txt_content = []
                for line in file_txt:
                    content = line.strip().split(" ")
                    txt_content.extend(content)
                cur_index = 0
                for item in txt_content:
                    txt_with_index_content.append((cur_index, item))
                    cur_index = cur_index + len(item) + 1
                # print(txt_with_index_content) 

                with open(os.path.join(folder_path, ann), 'r', encoding='utf-8') as file_ann:
                    for line in file_ann:
                        current_mes = line.strip().split("\t")
                        index_initial = current_mes[1].strip().split(" ")
                        if len(index_initial) == 3:
                            index = (int(index_initial[1]), int(index_initial[2]))
                        elif len(index_initial) == 4:
                            item_index = index_initial[2].find(";")
                            index = (int(index_initial[1]), int(index_initial[2][0:item_index]))
                        ann_label_list.append((index, current_mes[-1]))

                        ann_label_dict[index[0]] = (index[1], current_mes[-1])
                
                flag = False
                last_index = 0
                for item in txt_with_index_content:
                    men_index = item[0]
                    men_name = item[1]
                    if (flag == False) & (men_index in ann_label_dict.keys()): # matched! & B
                        BIO_taged.append(men_name)
                        BIO_taged.append("\t")
                        BIO_taged.append("B")
                        BIO_taged.append("\n")
                        flag = True
                        last_index = ann_label_dict[men_index][0]
                    elif flag == True: # matched! & I
                        if men_index <= last_index:
                            BIO_taged.append(men_name)
                            BIO_taged.append("\t")
                            BIO_taged.append("I")
                            BIO_taged.append("\n")
                        else: # end match & O
                            BIO_taged.append(men_name)
                            BIO_taged.append("\t")
                            BIO_taged.append("O")
                            BIO_taged.append("\n")
                            flag = False
                    else:
                        BIO_taged.append(men_name)
                        BIO_taged.append("\t")
                        BIO_taged.append("O")
                        BIO_taged.append("\n")
                
            taged.writelines(BIO_taged)
            taged.writelines(["\n"])      

    

if __name__ == "__main__":
    convert_files_to_txt("../../train_data", "../../preprocessed_data")
