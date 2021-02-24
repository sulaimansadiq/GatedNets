import torch


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk):

  maxk = max(topk)
  batch_size = target.size(0)

  target = torch.atleast_2d(target).t()
  # target_onehot = torch.cuda.FloatTensor(batch_size, 10)
  target_onehot = torch.zeros((batch_size, 10), device=output.device)
  # target_onehot.zero_()
  # target_onehot.to(device)
  target_onehot.scatter_(1, target, 1)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


