import copy


class early_stoping:
    def __init__(self, patience=5, early_stop_delta=0, restore_weight=True):
        self.patience = patience
        self.early_stop_delta = early_stop_delta
        self.restore_weight = restore_weight
        self.best_loss = None
        self.best_model = None
        self.counter = 0

    def __call__(self, net, valid_loss, stop = False):
        done = stop
        # generate initial state
        if self.best_loss == None:
            self.best_loss = valid_loss
            self.best_model = copy.deepcopy(net)

        # learning improved
        elif self.best_loss - valid_loss > self.early_stop_delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_model.load_state_dict(net.load_state_dict())

        # learning dit not improve
        elif self.best_loss - valid_loss < self.early_stop_delta:
            self.counter += 1
            if self.counter >= self.patience:
                done = True

        # restore waights
        if done:
            if self.restore_weight:
                net.load_state_dict(self.best_model.load_state_dict())
            return True

        return False

