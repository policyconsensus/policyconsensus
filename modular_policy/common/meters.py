import heapq


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg_ = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg_ = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg_ = self.sum / self.count

    def avg(self):
        return self.avg_
    

class TopkMeter:
    def __init__(self, k: int, smallest: bool = False):
        self.scalers = []
        self.k = k
        self.sign = -1 if smallest else 1

    def reset(self):
        self.scalers = []
    
    def update(self, val: int):
        if len(self.scalers) < self.k:
            heapq.heappush(self.scalers, self.sign * val)
        else:
            heapq.heappushpop(self.scalers, self.sign * val)

    def avg(self):
        return self.sign * sum(self.scalers) / len(self.scalers)
    
    def topk(self):
        return [self.sign * scaler for scaler in self.scalers]
    