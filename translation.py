import shutil
import warnings
import subprocess
from FeatureExtractor.extract_features import run as extract_features
from os.path import exists

def translate(video_path):
    
    lang = "pt"
    if not exists(video_path):
        return "Arquivo de vídeo não encontrado!"
    
    weight = './FeatureExtractor/checkpoints/archive/nslt_2000_065538_0.514762.pt'
    i3d_folder = './TSPNet/i3d-features'
    json = extract_features(weight, video_path, i3d_folder, 'rgb')
    
    f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w')
    f.write(json)
    f.close()
    
    f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w')
    f.write('Foo')
    f.close()
    
    os.chdir('./TSPNet/test_scripts')
    
    try:
        print("predicting.")
        warnings.filterwarnings("ignore")
        bashCommand = "bash test_phoenix_pos_embed_sp_test_3lvl.sh"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = str(output)[:-3][2:]
    except:
        print("An error occurred during prediction!")
        
    os.chdir('../..')
    shutil.rmtree('./TSPNet/i3d-features')
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w').close()
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w').close()
    
    return output