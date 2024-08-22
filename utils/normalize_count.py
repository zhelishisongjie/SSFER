import torch
def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    #print(num_batches)
    #print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5
    print(f"mean: {mean}")
    print(f"std: {std}")
    return mean,std

