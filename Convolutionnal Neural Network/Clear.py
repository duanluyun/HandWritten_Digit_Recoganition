import os
import shutil

def clear(tmp_path):
    filelist=[]
    dir=tmp_path
    filelist=os.listdir(dir)
    for f in filelist:
        filepath=os.path.join(dir,f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath,True)