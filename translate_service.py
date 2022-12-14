
import torch
from translation import translate as tspnet
from FeatureExtractor.models.pytorch_i3d import InceptionI3d

class TranslateService_TSPNet:

	def __init__(self, model_project_path):

		# model_project_path = './FeatureExtractor/checkpoints/archive/nslt_2000_065538_0.514762.pt'
		i3d = InceptionI3d(400, in_channels=3)
		i3d.replace_logits(2000)
		i3d.load_state_dict(torch.load(model_project_path)) # Network's Weight
		i3d.cuda()
		i3d.train(False)  # Set model to evaluate mode
		self.model = i3d

	def translate_from_video(self, video_path):

		# O MÉTODO TRANSLATE TEM QUE RODAR NA PASTA TSP-Prototype (ATENÇÃO TAMBÉM NO PATH DOS IMPORTS)
		# os.chdir("./path/to/TSP-Prototype/")

		return tspnet(video_path, self.model)

	def translate_from_keypoints(self, keypoints_data):
		raise NotImplementedError("Not implemented for this Model.")