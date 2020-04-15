import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginRankingLoss(nn.Module):
    def __init__(self, margin=None):
        super(TripletMarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_similarity = F.cosine_similarity(anchor, positive)
        negative_similarity = F.cosine_similarity(anchor, negative)
        labels = torch.ones(positive_similarity.size())
        if self.margin is None:
            diff = 1 - (torch.mean(positive_similarity) -
                        torch.mean(negative_similarity))
            margin = diff.item()
        else:
            margin = self.margin
        loss = F.margin_ranking_loss(positive_similarity, negative_similarity, labels.to(
            anchor.device), margin=margin)
        return loss
