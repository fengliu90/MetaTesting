import  torch
from    torch import nn
from    torch import optim
import numpy as np
from learner import Learner
from utils import MatConvert, MMDu

dtype = torch.float
device = torch.device("cuda:0")

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner(config).cuda()
        self.epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-8)), device, dtype)
        self.epsilonOPT.requires_grad = True
        self.sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.02), device, dtype)
        self.sigma0OPT.requires_grad = True
        self.meta_optim = optim.Adam(list(self.net.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT], lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry, is_training = True):
        """

        :param x_spt:   [b, setsz, d]
        :param y_spt:   [b, setsz, d]
        :param x_qry:   [b, querysz, d]
        :param y_qry:   [b, querysz, d]
        :return:
        """
        task_num, setsz, d = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        loss_not_train = torch.tensor(0.0)


        for i in range(task_num):
            # Get two samples
            S_spt = torch.cat((x_spt[i],y_spt[i]), 0).to(device, dtype)
            S_qry = torch.cat((x_qry[i], y_qry[i]), 0).to(device, dtype)

            # Initialize parameters
            ep = self.epsilonOPT ** 2
            sigma = self.sigmaOPT ** 2
            sigma0_u = self.sigma0OPT ** 2

            if is_training == False:
                return loss_not_train, self.net, sigma, sigma0_u, ep

            model_output = self.net(S_qry, self.net.parameters())
            TEMP = MMDu(model_output, querysz, S_qry, sigma, sigma0_u, ep)
            mmd_value_temp = -1 * TEMP[0]
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            loss = torch.div(mmd_value_temp, mmd_std_temp)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # first update
            with torch.no_grad():
                model_output_q = self.net(S_spt, self.net.parameters())
                TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                loss_q = torch.div(mmd_value_temp, mmd_std_temp)
                losses_q[0] += loss_q

            # second update
            with torch.no_grad():
                model_output_q = self.net(S_spt, fast_weights)
                TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                loss_q = torch.div(mmd_value_temp, mmd_std_temp)
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # run the i-th task and compute loss for k=1~K-1
                model_output = self.net(S_qry, fast_weights)
                TEMP = MMDu(model_output, querysz, S_qry, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                loss = torch.div(mmd_value_temp, mmd_std_temp)
                # loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
                # record the loss
                model_output_q = self.net(S_spt, fast_weights)
                TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                loss_q = torch.div(mmd_value_temp, mmd_std_temp)
                # print(loss_q.item())
                losses_q[k + 1] += loss_q

        print('sigma:',sigma.item(), 'sigma0:',sigma0_u.item(), 'epsilon:',ep.item())

        # sum over all losses across all tasks
        loss_q = losses_q[-1] / task_num
        print('J_value:',-loss_q.item())

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return -1 * loss_q, self.net, sigma, sigma0_u, ep
