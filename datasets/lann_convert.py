import os
import json

if __name__ == "__main__":
    directory = "/home/rs2926/workspace/Softname_Extraction/datasets/unconverted_files/gold_standard"
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for each_annotated in json_files:

        # --- extract structured information --- #
        sname_index_list = list() # sname index tuples in a list
        cur_file = os.path.join(directory, each_annotated)
        with open(cur_file, "r+") as f:
            file_dict = json.load(f)
            cur_keys = file_dict["indexes"].keys()
            for cur_key in cur_keys:
                cur_index_info = file_dict["indexes"][cur_key]
                if ("Entity" in cur_index_info.keys()) and (len(cur_index_info["Entity"])!=0):
                    sname_index_list.append((cur_index_info["Entity"][0]["begin"], cur_index_info["Entity"][0]["end"]))
            
            # --- index --- 
            # for start_index, end_index in sname_index_list:
                # print(file_dict["content"][start_index: end_index])
            # --- name ---
            # file_dict["content"][start_index: end_index]
            """
            T1	software 0 41	IBIS integrated biological imaging system
            T2	software 103 120	Unix workstations
            T3	software 125 129	IBIS
            T4  software 368 385    Unix workstations
            """
            
            file_name_without_extension = os.path.splitext(each_annotated)[0]
            txt_file_path = '../train_data/' + file_name_without_extension + '.txt'
            ann_file_path = '../train_data/' + file_name_without_extension + '.ann'

            # 写入文本文件
            with open(txt_file_path, 'w') as txt_file:
                print(each_annotated)
                txt_file.write(file_dict["content"])
            # exit()
            # 写入 ann 文件
            # with open(ann_file_path, 'w') as ann_file:
                # ann_file.write("This is an example of writing to an .ann file.\n")
                # ann_file.write("Annotation data can be added here.\n")

        

        

