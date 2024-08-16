def find_sublist(main_list, sub_list):
    sub_len = len(sub_list)
    for i in range(len(main_list) - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i, i+sub_len-1
    return -1  # 如果没有找到子列表，返回 -1

main_list = [1, 2, 3, 4, 5, 6]
sub_list = [1,2,3,4]

first, end = find_sublist(main_list, sub_list)
print(first, end)