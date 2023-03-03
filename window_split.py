import torch

def get_batch(data, targets, time_step, skip=1, isTrain=False):
    """
        data: size([n_features, data_len])
        targets: size([data_len, n_classes])
    """
    data = data.unsqueeze(0)
    inp = torch.tensor([])
    start = 0
    end = 0
    if isTrain == True:
        n = data.size(2) - time_step + 1 - skip
        for i in range(n):
            end = start + time_step
            inp = torch.cat((inp, data[:, :, start:end]))
            start += 1
        out = targets[time_step+skip-1:time_step+skip+n-1, :]
    else:
        n = data.size(2) - time_step + 1 - skip
        for i in range(n):
            end = start + time_step
            inp = torch.cat((inp, data[:, :, start:end]))
            start += 1
        out = targets[time_step+skip-1:, :]
    return inp, out
