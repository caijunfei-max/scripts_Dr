def xdatcar_extract(file_path, atoms_num):
    with open(file_path, "r", encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    
    data_confi = []
    for i in data:
        if "Direct configuration" in i:
            index_i = data.index(i)
            a = index_i+1
            b = a+atoms_num
            confi_i = data[a: b]
            data_confi.append(confi_i)
    return data_confi


def xdatcar_splice(path_list, atoms_num):
    # 输入数据应该是一个列表，里面包括所有的xdatcar的路径，仅限于同体系的拼接，不同系统的得到的是个错误的xdatcar
    output = []
    with open(path_list[0], 'r', encoding='utf-8') as f:
        for i in range(7):
            line = f.readline()
            output.append(line)
        
    confi=[]
    for i in path_list:
        info = xdatcar_extract(i, atoms_num)
        for j in info:
            confi.append(j)
    for i in confi:
        configuration_num = confi.index(i)+1
        xdatcar_str = "Direct configuration=    {0}\n".format(str(configuration_num))
        output.append(xdatcar_str)
        output.append(i)
        
    return output

path= ["XDATCAR_1", "XDATCAR_2"]

with open("XDATCAR_COM", "w") as file:
    for item in xdatcar_splice(path, 96):
        print("itering")
        if type(item) == str:
            file.write(item)
        elif type(item) == list:
            for data in item:
                file.write(data)
    file.close()
