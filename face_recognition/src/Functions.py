import matplotlib.pyplot as plt
import torch
import tqdm

def fine_tune(model, dataloader, optimizer, criterion, lr_scheduler,
              device, start_epoch, last_epoch, val_dataloader, epoch_loss_dynamic):
    for epoch in range(start_epoch, last_epoch + 1):
        model.train()
        epoch_loss = 0
        tqdm_dataloader = tqdm.tqdm(dataloader)
        for i, (images, labels) in enumerate(tqdm_dataloader):
            images = images.to(device) # type: ignore
            labels = labels.to(device) # type: ignore
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                tqdm_dataloader.set_postfix(loss=epoch_loss / i)
            if (i + 1) % 500 == 0:
                epoch_loss_dynamic.append(epoch_loss / i)
                save_plot(epoch_loss_dynamic)
        lr_scheduler.step()
        save_checkpoint(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_scheduler=lr_scheduler,
                        index=epoch,
                        epoch_loss_dynamic=epoch_loss_dynamic)
        accuracy = validate(model=model,
                            val_dataloader=val_dataloader,
                            device=device)
        print(f'Epoch: {epoch}/{last_epoch}')
        print(f'Accuracy: {accuracy:.2f}%')
        print()
    return epoch_loss_dynamic

def save_plot(epoch_loss_dynamic):
    plt.figure(figsize=(10, 5), dpi=200)
    plt.title('Loss Dynamic')
    plt.xlabel('Steps (x500)')
    plt.ylabel('Loss')
    plt.plot(epoch_loss_dynamic)
    plt.grid(True)
    plt.savefig('./documentation/loss_dynamic.png')
    plt.close()

def save_checkpoint(model, optimizer, criterion, lr_scheduler, epoch_loss_dynamic, index):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch_loss_dynamic': epoch_loss_dynamic
    }
    torch.save(checkpoint, f'./checkpoints/checkpoint{index}.pth')

def validate(model, val_dataloader, device, threshold=0.3):
    model.eval()
    correct = 0
    total = 0
    tqdm_val_dataloader = tqdm.tqdm(val_dataloader)
    with torch.no_grad():
        for img1, img2, label in tqdm_val_dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            output1 = model(img1)
            output2 = model(img2)
            similarity = (output1 * output2).sum(dim=1)
            distance = 1 - similarity
            preds = (distance < threshold).long()
            correct += (preds == label).sum().item() # type: ignore
            total += label.size(0)
    return correct / total * 100