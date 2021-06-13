import numpy as np

class SynDataMetaTST:

    def __init__(self, data_TST, batchsz, n_way, k_shot, k_query):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        """

        self.x = data_TST
        num_meta_tasks = self.x.shape[0]
        split_ratio = 0.8
        numM_train = int(num_meta_tasks * split_ratio)
        self.x_train, self.x_test = self.x[:numM_train], self.x[numM_train:]

        # Fetch parameters
        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]
        self.N = self.x.shape[1]
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        assert (k_shot + k_query) <=self.N

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 200, 2]
        :return: A list with [tr_set_x, tr_set_y, te_x, te_y] ready to be fed to our networks
        """
        #  take 2 way 50 shot as example: 2 * 50
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []
        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way*2, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(int(self.N/2), int((self.k_shot + self.k_query)/2), False)

                    if j % 2 == 0:
                        x_spt.append(data_pack[cur_class][selected_img[:int(self.k_shot / 2)]])
                        x_qry.append(data_pack[cur_class][selected_img[int(self.k_shot / 2):]])
                        y_spt.append(data_pack[cur_class][int(self.N / 2) + selected_img[:int(self.k_shot / 2)]])
                        y_qry.append(data_pack[cur_class][int(self.N / 2) + selected_img[int(self.k_shot / 2):]])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * int(self.k_shot/2))
                x_spt = np.array(x_spt).reshape(self.n_way * int(self.k_shot/2), 2)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * int(self.k_shot/2), 2)[perm]
                perm = np.random.permutation(self.n_way * int(self.k_query/2))
                x_qry = np.array(x_qry).reshape(self.n_way * int(self.k_query/2), 2)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * int(self.k_query/2), 2)[perm]

                # append tasks
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # generate batch
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, int(setsz/2), 2)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batchsz, int(setsz/2), 2)
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, int(querysz/2), 2)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batchsz, int(querysz/2), 2)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
