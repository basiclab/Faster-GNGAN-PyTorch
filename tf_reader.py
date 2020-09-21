import numpy as np
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


if __name__ == '__main__':
    path = "./logs/BIGGAN/CIFAR10/0_GN_2/events.out.tfevents.1600494159.yilun-lab2"

    # Loading too much data is slow...
    tf_size_guidance = {
        'scalars': 0,
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())

    loss = event_acc.Scalars('loss')
    loss_real = event_acc.Scalars('loss_real')
    loss_fake = event_acc.Scalars('loss_fake')

    with tqdm(loss, desc='loss') as pbar:
        for event in pbar:
            if np.isnan(event.value) or np.isinf(event.value):
                pbar.write(str(event.step) + ", " + str(event.value))
                break
    with tqdm(loss_real, desc='loss_real') as pbar:
        for event in pbar:
            if np.isnan(event.value) or np.isinf(event.value):
                pbar.write(str(event.step) + ", " + str(event.value))
                break
    with tqdm(loss_fake, desc='loss_fake') as pbar:
        for event in pbar:
            if np.isnan(event.value) or np.isinf(event.value):
                pbar.write(str(event.step) + ", " + str(event.value))
                break
