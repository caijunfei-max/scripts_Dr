
serialnum = []
with open("mp-id.txt","r") as f1:
    for ip in f1.readlines():
        if ip != None:
            serialnum.append(ip.strip("\n"))
f1.close()
 #阅读某个文件中的每一行的数据到一个列表中

