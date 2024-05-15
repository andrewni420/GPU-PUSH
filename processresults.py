import os 

def count_lines():
    nlines = 0
    ntestlines=0
    nfiles = 0
    ntestfiles=0
    for (dir_path,dir_names,file_names) in os.walk("src"):
        for f in [f for f in file_names if f[-3:]==".py"]:
            with open(f"{dir_path}/{f}") as file:
                nlines+=len(file.readlines())
                nfiles+=1

    for (dir_path,dir_names,file_names) in os.walk("tests"):
        for f in [f for f in file_names if f[-3:]==".py"]:
            with open(f"{dir_path}/{f}") as file:
                ntestlines+=len(file.readlines())
                ntestfiles+=1
    print(f"NLINES {nlines} NFILES {nfiles}\nNTESTLINES {ntestlines} NTESTFILES {ntestfiles} \nTOTAL {nlines+ntestlines} FILES {ntestfiles+nfiles}")

count_lines()