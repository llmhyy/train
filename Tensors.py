class Tensors:
    def __init__(self, training_tensor, cost_tensor, prob_tensor, xs, ys,
                 ws1, bs1, ws2, bs2):
        self.training_tensor = training_tensor
        self.cost_tensor = cost_tensor
        self.prob_tensor = prob_tensor
        self.xs = xs
        self.ys = ys
        self.ws1 = ws1
        self.ws2 = ws2
        self.bs1 = bs1
        self.bs2 = bs2
