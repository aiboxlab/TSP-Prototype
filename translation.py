import os
import shutil
import subprocess
from FeatureExtractor.extract_features import run as extract_features
from os.path import exists

def translate(input_path):
    # Must be on TSP-Prototype Folder
    lang = "pt"
    if not exists(input_path):
        print(input_path,": Video file was not found!")
        return ""
    
    try:
        weight = './FeatureExtractor/checkpoints/archive/nslt_2000_065538_0.514762.pt'
        i3d_folder = './TSPNet/i3d-features'
        json = extract_features(weight, input_path, i3d_folder, 'rgb')

        f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w')
        f.write(json)
        f.close()

        f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w')
        f.write('Foo')
        f.close()
    except:
        print(input_path,": An error occurred during extraction!")
        return ""
    
    output = ""

    try:
        print("predicting.")
        bashCommand = "bash ./TSPNet/test_scripts/test_phoenix_pos_embed_sp_test_3lvl.sh"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        output, error = process.communicate()
        with open('./output.txt') as f:
            output = f.readlines()[0]
    except:
        print(input_path,": An error occurred during prediction!")
        
    shutil.rmtree('./TSPNet/i3d-features')
    open('./output.txt', 'w').close()
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w').close()
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w').close()
    
    return output