import  torch
import  numpy as np
from dataTST import SynDataMetaTST
from MetaTST import Meta
import argparse
parser = argparse.ArgumentParser()
from utils import MatConvert, TST_MMD_u, MMDu

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    dtype = torch.float
    device = torch.device("cuda:0")

    d = args.d  # dimension of data
    n = args.n  # number of samples in per mode
    n_te = args.n_te # number of training samples for the target task
    K = args.K  # number of trails
    num_meta_tasks = args.num_meta_tasks # number of meta-samples
    print('n: ' + str(n) + ' d: ' + str(d))

    N_per = 100  # permutation times
    alpha = 0.05  # test threshold
    x_in = d  # number of neurons in the input layer, i.e., dimension of data
    H = 30  # number of neurons in the hidden layer
    x_out = 3 * d  # number of neurons in the output layer
    learning_rate = 0.00005 # learning rate for MMD-D
    N_epoch = 1000  # maximim number of epochs for training
    N = 100  # # number of test sets
    N_f = 100.0  # number of test sets (float)
    list_nte = [50, 80, 100, 120, 150, 200, 250] # number of test samples for the target task

    config = [
        ('linear', [H, x_in]),
        ('softplus', [True]),
        ('linear', [H, H]),
        ('softplus', [True]),
        ('linear', [H, H]),
        ('softplus', [True]),
        ('linear', [x_out, H]),
    ]

    print(args)

    # Generate variance and co-variance matrix of Q (target task)
    Num_clusters = 2
    mu_mx = np.zeros([Num_clusters, d])
    mu_mx[1] = mu_mx[1] + 0.5
    sigma_mx_1 = np.identity(d)
    sigma_mx_2 = [np.identity(d), np.identity(d)]
    sigma_mx_2[0][0, 1] = 0.7
    sigma_mx_2[0][1, 0] = 0.7
    sigma_mx_2[1][0, 1] = -0.7
    sigma_mx_2[1][1, 0] = -0.7

    mu_mx_s = np.zeros([Num_clusters, d])
    mu_mx_s[1] = mu_mx_s[1] + 0.5
    sigma_mx_1_s = np.identity(d)
    sigma_mx_2_s = [np.identity(d), np.identity(d)]

    # Naming variables
    s1 = np.zeros([n * Num_clusters, d])
    s2 = np.zeros([n * Num_clusters, d])
    s1_te = np.zeros([n_te * Num_clusters, d])
    s2_te = np.zeros([n_te * Num_clusters, d])
    J_star_u = np.zeros([N_epoch])
    J_star_u_init = np.zeros([N_epoch])
    Results = np.zeros([len(list_nte), 2, K])

    # repeat experiments K times
    for kk in range(K):

        maml = Meta(args, config).to(device)
        maml_init = Meta(args, config).to(device)

        tmp = filter(lambda x: x.requires_grad, maml.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        # print(maml)
        print('Total trainable tensors:', num)

        tmp = filter(lambda x: x.requires_grad, maml_init.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        # print(maml_init)
        print('Total trainable tensors:', num)

        # generate meta-samples
        data_org = np.random.randn(num_meta_tasks,4*n,2)
        for nn in range(num_meta_tasks):
            sigma_mx_2_s[0][0, 1] = 0.6 - args.closeness + (0.1/num_meta_tasks) * (nn+1)
            sigma_mx_2_s[0][1, 0] = 0.6 - args.closeness + (0.1/num_meta_tasks) * (nn+1)
            sigma_mx_2_s[1][0, 1] = -(0.6 - args.closeness + (0.1/num_meta_tasks) * (nn+1))
            sigma_mx_2_s[1][1, 0] = -(0.6 - args.closeness + (0.1/num_meta_tasks) * (nn+1))

            for i in range(Num_clusters):
                np.random.seed(seed=1102*nn + i + n)
                s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1_s, n)
            for i in range(Num_clusters):
                np.random.seed(seed=819*nn + 1 + i + n)
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2_s[i], n)
            data_org[nn] = np.concatenate((s1, s2), axis=0)
        # get training loader for meta-samples
        db_train = SynDataMetaTST(data_org, 10, 2, 150, 50)

        # train meta kernels using the generated training loader
        for step in range(args.epoch):

            x_spt, y_spt, x_qry, y_qry = db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # get the same init parameters for MMD-D
            J_value_init, model_u_init, sigma_init, sigma0_u_init, ep_init = maml_init(x_spt, y_spt, x_qry, y_qry, is_training=False)
            # train meta kernels
            J_value, model_u, sigma, sigma0_u, ep = maml(x_spt, y_spt, x_qry, y_qry)
            # print objectives from epoch
            if step % 10 == 0:
                print('step:', step, '\ttraining J value:', J_value.item())

        # setup meta kernels
        torch.manual_seed(1 * 19 + n)
        torch.cuda.manual_seed(kk * 19 + n)
        epsilonOPT = MatConvert(np.ones(1) * np.sqrt(ep.detach().cpu().numpy()), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(sigma.detach().cpu().numpy()), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(sigma0_u.detach().cpu().numpy()), device, dtype)
        sigma0OPT.requires_grad = True
        print(epsilonOPT.item())

        # setup optimizer for training deep kernel
        optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                       lr=learning_rate/3)

        # setup init Kernels
        torch.manual_seed(1 * 19 + n)
        torch.cuda.manual_seed(kk * 19 + n)
        epsilonOPT_init = MatConvert(np.ones(1) * np.sqrt(ep_init.detach().cpu().numpy()), device, dtype)
        epsilonOPT_init.requires_grad = True
        sigmaOPT_init = MatConvert(np.ones(1) * np.sqrt(sigma_init.detach().cpu().numpy()), device, dtype)
        sigmaOPT_init.requires_grad = True
        sigma0OPT_init = MatConvert(np.ones(1) * np.sqrt(sigma0_u_init.detach().cpu().numpy()), device, dtype)
        sigma0OPT_init.requires_grad = True
        print(epsilonOPT_init.item())

        # Setup optimizer for training init kernel
        optimizer_u_init = torch.optim.Adam(list(model_u_init.parameters()) + [epsilonOPT_init] + [sigmaOPT_init] + [sigma0OPT_init],
                                       lr=learning_rate)


        # Generate training data for target tasks
        for i in range(Num_clusters):
            np.random.seed(seed=1102*kk + i + n)
            s1_te[n_te * (i):n_te * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n_te)
        for i in range(Num_clusters):
            np.random.seed(seed=819*kk + 1 + i + n)
            s2_te[n_te * (i):n_te * (i + 1), :] = np.random.multivariate_normal(mu_mx_s[i], sigma_mx_2[i], n_te)
        S = np.concatenate((s1_te, s2_te), axis=0)
        S = MatConvert(S, device, dtype)
        N1 = Num_clusters*n_te

        # Meta Kernels as init when training with training set from the target task
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(50):
            # one way to train kernel with limited data
            n_random = int(n_te*Num_clusters/5)
            selected_cls1 = np.random.choice(n_te * Num_clusters, n_random, False)
            selected_cls2 = np.random.choice(n_te * Num_clusters, n_random, False)
            s1_te_random = s1_te[selected_cls1,:]
            s2_te_random = s2_te[selected_cls2, :]
            S_random = np.concatenate((s1_te_random, s2_te_random), axis=0)
            S_random = MatConvert(S_random, device, dtype)
            # another way to train kernel with limited data (similar performance)
            # S_random = S
            # n_random = N1

            # Compute epsilon, sigma and sigma_0
            ep = epsilonOPT ** 2
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2

            # Compute output of the deep network
            modelu_output = model_u(S_random)

            # Compute J (STAT_u)
            TEMP = MMDu(modelu_output, n_random, S_random, sigma, sigma0_u, ep)
            mmd_value_temp = -1 * (TEMP[0] + 10 ** (-8))
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            if mmd_std_temp.item() == 0:
                print('error!!')
            if np.isnan(mmd_std_temp.item()):
                print('error!!')
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            J_star_u[t] = STAT_u.item()

            # Initialize optimizer and Compute gradient
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)

            # Update weights using gradient descent
            optimizer_u.step()
            # Print MMD, std of MMD and J
            if t % 100 == 0:
                print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                      -1 * STAT_u.item())

        # random kernel as init when training with training set from the target task --
        # --> validate the consistence of performance of MMD-D
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(1000):
            # Compute epsilon, sigma and sigma_0
            ep_init = epsilonOPT_init ** 2
            sigma_init = sigmaOPT_init ** 2
            sigma0_u_init = sigma0OPT_init ** 2
            # Compute output of the deep network
            modelu_output_init = model_u_init(S)
            # Compute J (STAT_u)
            TEMP_init = MMDu(modelu_output_init, N1, S, sigma_init, sigma0_u_init, ep_init)
            mmd_value_temp_init = -1 * (TEMP_init[0] + 10 ** (-8))
            mmd_std_temp_init = torch.sqrt(TEMP_init[1] + 10 ** (-8))
            if mmd_std_temp_init.item() == 0:
                print('error!!')
            if np.isnan(mmd_std_temp_init.item()):
                print('error!!')
            STAT_u_init = torch.div(mmd_value_temp_init, mmd_std_temp_init)
            # STAT_u = mmd_value_temp # D+M
            J_star_u_init[t] = STAT_u_init.item()
            # Initialize optimizer and Compute gradient
            optimizer_u_init.zero_grad()
            STAT_u_init.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u_init.step()
            # Print MMD, std of MMD and J
            if t % 100 == 0:
                print("mmd_value_init: ", -1 * mmd_value_temp_init.item(), "mmd_std_init: ", mmd_std_temp_init.item(), "Statistic_init: ",
                      -1 * STAT_u_init.item())

        # test the trained kernel on the target task (with different sample size: 50, 80, 100, 120, 150, 200, 250)
        for i_test in range(len(list_nte)):
            n_te2 = list_nte[i_test]
            s1_te2 = np.zeros([n_te2 * Num_clusters, d])
            s2_te2 = np.zeros([n_te2 * Num_clusters, d])
            N1_te2 = Num_clusters * n_te2
            H_u = np.zeros(N)
            T_u = np.zeros(N)
            M_u = np.zeros(N)
            H_u_init = np.zeros(N)
            T_u_init = np.zeros(N)
            M_u_init = np.zeros(N)
            np.random.seed(1102)
            count_u = 0
            count_u_init = 0
            for k in range(N):
                # Generate target tasks
                for i in range(Num_clusters):
                    np.random.seed(seed=1102 * (k+2) + 2*kk + i + n)
                    s1_te2[n_te2 * (i):n_te2 * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n_te2)
                for i in range(Num_clusters):
                    np.random.seed(seed=819 * (k + 1) + 2*kk + i + n)
                    s2_te2[n_te2 * (i):n_te2 * (i + 1), :] = np.random.multivariate_normal(mu_mx_s[i], sigma_mx_2[i], n_te2)
                S = np.concatenate((s1_te2, s2_te2), axis=0)
                S = MatConvert(S, device, dtype)

                # Run two sample test (deep kernel) on generated data
                h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1_te2, S, sigma, sigma0_u, ep, alpha, device, dtype)
                h_u_init, threshold_u_init, mmd_value_u_init = TST_MMD_u(model_u_init(S), N_per, N1_te2, S, sigma_init, sigma0_u_init, ep_init, alpha, device, dtype)

                # Gather results
                count_u = count_u + h_u
                count_u_init = count_u_init + h_u_init
                print("Meta_KL:", count_u, "MMD-DK:", count_u_init)
                H_u[k] = h_u
                T_u[k] = threshold_u
                M_u[k] = mmd_value_u
                H_u_init[k] = h_u_init
                T_u_init[k] = threshold_u_init
                M_u_init[k] = mmd_value_u_init

            # Print test power of MetaKL and MMD-D
            print("Test Power of Meta MMD: ", H_u.sum() / N_f)
            Results[i_test, 0, kk] = H_u.sum() / N_f
            print("Test Power of Meta MMD (K times): ", Results[i_test, 0])
            print("Average Test Power of Meta MMD: ", Results[i_test, 0].sum() / (kk + 1))

            print("Test Power of deep MMD: ", H_u_init.sum() / N_f)
            Results[i_test, 1, kk] = H_u_init.sum() / N_f
            print("Test Power of deep MMD (K times): ", Results[i_test, 1])
            print("Average Test Power of deep MMD: ", Results[i_test, 1].sum() / (kk + 1))

        print(Results[:,:,kk])


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int, default=50)  # n_i per mode for each task
    argparser.add_argument('--n_te', type=int, default=150)  # number of training samples per mode for target task
    argparser.add_argument('--d', type=int, default=2)  # dimension of samples
    argparser.add_argument('--K', type=int, default=10)  # number of trails num_meta_tasks
    argparser.add_argument('--num_meta_tasks', type=int, default=100) # number of meta-samples
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000) # training epochs for training meta kernels
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.8)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--closeness', type=float, default=0.3)

    args = argparser.parse_args()

    main(args)
