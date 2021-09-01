# -*- coding: utf-8 -*

import os
import string
from PIL import Image
import json
import sys
import tempfile

class test_detector:

    @classmethod
    def readcmd(cls, cmd):
        try:
            ftmp = tempfile.NamedTemporaryFile(suffix='.out', prefix='tmp', delete=False)
            fpath = ftmp.name
            if os.name=="nt":
                fpath = fpath.replace("/","\\") # forwin
            ftmp.close()
            os.system(cmd + " > " + fpath)
            data = ""
            with open(fpath, 'r') as file:
                data = file.read()
                file.close()
            os.remove(fpath)
            return data
        except:
            print(sys.exc_info()[0])

    def __init__(self):
        pass


    @classmethod
    def test(cls, img, path, graph, session):
        try:
            #path = os.path.abspath(os.getcwd())
            #print(path)
            img.save(os.path.abspath(os.getcwd()) + '/original.' + img.format.lower())

            modelname = ''
            for filename in os.listdir(path):
                if filename.endswith('.hdf5'):
                    modelname = os.path.splitext(filename)[0]
            #print('\n\n' + modelname)
            fin = open(path + modelname + '.txt', 'rt')
            data = fin.read()
            fin.close()
            resultstr = cls.readcmd('python3 CrnnUtility_cpu.py ' + modelname + ' original.' + img.format.lower() + ' Tsai-Hsing.Lu@quantatw.com ' + data.replace('\n','') + ' log_' + modelname + '.txt  2>&1 | tee -a ./temp.log')
            #print('\n\n\n\n\n\n\n')
            #print(resultstr.split('\n'))
            resultarr = resultstr.split('\n')
            returnarr = []
            for str in resultarr:
                if str.startswith('[') and str.endswith(']'):
                    #print(str + '\n')
                    tmparr = str.replace("['",'').replace("']",'').split(' ')
                    tmpval = {
                        'value':tmparr[0]
                    }
                    returnarr.append(tmpval)
            returnval = {
                    'isSuccess':'true',
                    'ErrorMsg':'',
                    'result':returnarr
            }
            return returnval
        except:
            returnval = {
                    'isSuccess': 'false',
                    'ErrorMsg': sys.exc_info()[0],
            }
            return json.dumps(returnval)


