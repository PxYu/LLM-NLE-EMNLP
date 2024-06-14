import torch
import torch.nn.functional as F

class CalibratedListwiseSoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(CalibratedListwiseSoftmaxLoss, self).__init__()

        # Define the learnable parameters y_0 and s_0
        self.y_0 = torch.nn.Parameter(torch.zeros(1, 1))
        self.s_0 = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, S_q, Y_q):
        # S_q, Y_q -> batch_size, 1
        # Add y_0 and s_0 to Y_q and S_q
        S_q = torch.cat([self.s_0, S_q], dim=0)
        Y_q = torch.cat([self.y_0, Y_q.float()], dim=0)

        # Calculate the softmax loss
        softmax_target = F.softmax(Y_q, dim=0)
        log_softmax_prediction = F.log_softmax(S_q, dim=0)
        listwise_cross_entropy = torch.mean(-softmax_target * log_softmax_prediction)
        return listwise_cross_entropy