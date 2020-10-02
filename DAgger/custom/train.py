import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from custom.model import CustomModel, load_model
from custom.early_stopping import EarlyStopping

def train(model_name, train_loader, val_loader, epochs = 100):
    f = CustomModel()
    if torch.cuda.is_available():
        f = f.cuda()

    lr_lambda = lambda epoch: 1 - 0.02*(epoch - 50) if epoch > 49 else 1
    optimizer = Adam(f.parameters())
    scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)

    start_epoch = load_model(model_name, f)
    es = EarlyStopping(patience=5)
    for epoch in range(start_epoch, epochs):
        f.train()
        start = time.process_time()
        for im, (steering, throttle) in train_loader:
            if torch.cuda.is_available():
                im = im.cuda()
                steering = steering.cuda()
                throttle = throttle.cuda()
            
            optimizer.zero_grad()
            psteering, pthrottle = f(im)

            loss = nn.functional.mse_loss(psteering, steering) + nn.functional.mse_loss(pthrottle, throttle)
            loss.backward()

            optimizer.step()

        scheduler.step()
        # val
        with torch.no_grad():
            f.eval()
            val_loss = torch.tensor(0.0)
            for im, (steering, throttle) in train_loader:
                if torch.cuda.is_available():
                    im = im.cuda()
                    steering = steering.cuda()
                    throttle = throttle.cuda()

                psteering, pthrottle = f(im)
                val_loss += nn.functional.mse_loss(psteering, steering) + nn.functional.mse_loss(pthrottle, throttle)
            
            print(f'val_loss: {val_loss.item()}')
            if es.step(val_loss):
                break

        # save after each epoch
        torch.save(f.state_dict(), f'models/{model_name}.pt')
        print(f'ep{epoch} in {time.process_time() - start:.0f}s.')