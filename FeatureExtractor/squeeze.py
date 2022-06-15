import os
import torch
import pickle
import torch.nn as nn

# ==== AVG POOL ====
class POOL(nn.Module):
    def __init__(self):
        super(POOL, self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(2, stride=1), # nn.AvgPool2d(2, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


def squeeze_net(name, features, stride):

    model = POOL()
    video_mov = name + ".mov"

    # ===== Open pickle file =====
    data = []
    with open('./FeatureExtractor/norm_dataset_fa.pkl', 'rb') as f:
        data = pickle.load(f)

    if video_mov in data.keys():
        print("squeezing.")
        obj = data[video_mov]
        spanned_list = []
        for i in range(len(obj)):
            if i % stride == 0:
                spanned_list.append(obj[i])

        unified = []
        for i in range(len(features)): # para cada i de features
            tensor = features[i][0]

            pkl_tensor = []
            if i >= len(spanned_list):
                pkl_tensor = spanned_list[-1]
            else:
                pkl_tensor = torch.tensor(spanned_list[i])

            merged = torch.stack([ tensor[:960] , pkl_tensor ])
            merged = torch.unsqueeze(merged, 0)
            pooled = model(merged)

            array = pooled[0][0]

            concatenated = torch.cat((array, tensor[960:]))
            concatenated = torch.cat((concatenated, torch.tensor([0])))

            result = torch.unsqueeze(concatenated, 0)

            unified.append(result)

        features = unified

    else:
        print("NAME ISNT IN KEYS!")

    return features