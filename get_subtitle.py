import shutil
import warnings
import subprocess
from FeatureExtractor.extract_features import run

def get_subtitle(video_path, lang):
    weight = './FeatureExtractor/checkpoints/archive/nslt_2000_065538_0.514762.pt'
    out = './TSPNet/i3d-features'
    json = run(weight, video_path, out, 'rgb')
    
    f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w')
    f.write(json)
    f.close()
    
    f = open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w')
    f.write('Foo')
    f.close()
    
    os.chdir('./TSPNet/test_scripts')
    
    print("predicting.")
    warnings.filterwarnings("ignore")
    bashCommand = "bash test_phoenix_pos_embed_sp_test_3lvl.sh"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    shutil.rmtree('../i3d-features')
    
    os.chdir('../..')
    
    with open('./TSPNet/output.txt') as f:
        output = f.readlines()[0]

    open('./TSPNet/output.txt', 'w').close()
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.sign', 'w').close()
    open('./TSPNet/data-bin/phoenix2014T/sp25000/test.sign-'+lang+'.'+lang, 'w').close()
    
    return output