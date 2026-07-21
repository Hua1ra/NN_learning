import torch
import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

# Function to validate model's confidence accuracy
def validate(data_loader, model, device):
    with torch.no_grad():
        model.eval()
        pos_confidence, neg_confidence = 0.0, 0.0
        pos_count, neg_count = 0, 0
        # For each image we calculate predictions
        # For each prediction we calculate confidence accuracy
        for images, labels in tqdm.tqdm(data_loader):
            images = images.to(device)
            labels = labels.view(-1, 20 * 20, 5).to(device)
            predictions = torch.sigmoid(model(images))[:, :, 0]
            labels_coords_mask = (labels[:, :, 0] != 0).bool().to(device)
            # There can be 0 signs on the image. Need to check to prevent NaN
            if labels_coords_mask.sum() != 0:
                # Sum of the confidences across positive pixels
                pos_confidence += predictions[labels_coords_mask].sum().item()
                pos_count += labels_coords_mask.sum().item()
            # Sum of the confidences across negative pixels
            neg_confidence += predictions[~labels_coords_mask].sum().item()
            neg_count += (~labels_coords_mask).sum().item()
    return pos_confidence / (pos_count + 1e-6), neg_confidence / (neg_count + 1e-6)

# Function to train the model for a single epoch
def train_epoch(data_loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    tqdm_loader = tqdm.tqdm(data_loader)
    # Standart learning cycle
    for i, (images, labels) in enumerate(tqdm_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # For each 1000's iteration, we'll see the average loss
        if (i + 1) % 100 == 0:
            tqdm_loader.set_postfix(loss=f'{running_loss / (i + 1):.4f}')
    return running_loss / len(data_loader)

# Function to train the model from last_epoch to num_epochs
def train(data_loader, model, optimizer, criterion, scheduler, num_epochs, last_epoch):
    # For each epoch perform learning and validation
    for epoch in range(last_epoch, num_epochs):
        # Train for 1 epoch
        avg_loss = train_epoch(data_loader, model, optimizer, criterion)
        scheduler.step()
        # Get validation metrics
        pos_confidence, neg_confidence = validate(data_loader, model)
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print(f'Average loss: {avg_loss:.4f}')
        print(f'Positive confidence: {pos_confidence:.4f}')
        print(f'Negative confidence: {neg_confidence:.4f}')
        print()
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'num_epochs': num_epochs
        }
        torch.save(checkpoint, f'./resources/models/model{epoch + 1}.pth')

# Intersaction over Union metric
def iou(bbox1, bbox2, size): # bbox=[p, x, y, w, h, i, j]
    # Convert to bbox coordinates to relative
    x1_center = int((bbox1[6] + bbox1[1]) * (size[0] / 20))
    y1_center = int((bbox1[5] + bbox1[2]) * (size[1] / 20))
    x1_min = int(x1_center - bbox1[3] * size[0] / 20)
    x1_max = int(x1_center + bbox1[3] * size[0] / 20)
    y1_min = int(y1_center - bbox1[4] * size[1] / 20)
    y1_max = int(y1_center + bbox1[4] * size[1] / 20)
    x2_center = int((bbox2[6] + bbox2[1]) * (size[0] / 20))
    y2_center = int((bbox2[5] + bbox2[2]) * (size[1] / 20))
    x2_min = int(x2_center - bbox2[3] * size[0] / 20)
    x2_max = int(x2_center + bbox2[3] * size[0] / 20)
    y2_min = int(y2_center - bbox2[4] * size[1] / 20)
    y2_max = int(y2_center + bbox2[4] * size[1] / 20)

    # Get inetrsaction relative coordinates and area
    intersect_xmin = max(x1_min, x2_min)
    intersect_ymin = max(y1_min, y2_min)
    intersect_xmax = min(x1_max, x2_max)
    intersect_ymax = min(y1_max, y2_max)
    intersection = (max(intersect_xmax - intersect_xmin, 0)) * (max(intersect_ymax - intersect_ymin, 0))
    # Get union area
    union = ((x1_max - x1_min) * (y1_max - y1_min) +
             (x2_max - x2_min)  * (y2_max - y2_min) -
             intersection)
    return intersection / union if union > 0 else 0

# Get image with boxes (use Non-Maximum Suppression for prediciotns)
def get_detected(model,
                 device,
                 img_path='./resources/tt100k_2021/train/23.jpg',
                 transformer=None,
                 threshold=0.15,
                 iou_threshold=0.1):
    model.eval()
    with torch.no_grad():
        # Define transformer to feedd into model
        if transformer is None:
            transformer = transforms.Compose([
                            transforms.Resize((640, 640)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                        ])
        img = Image.open(img_path).convert('RGB')
        # Get predictions
        img_tensor = transformer(img).unsqueeze(0).to(device)
        predictions = model(img_tensor).view(20, 20, 5)
        predictions = torch.sigmoid(predictions).cpu().numpy()
        boxes = []
        # Select only valid predictions
        for i in range(20):
            for j in range(20):
                if predictions[i, j, 0] >= threshold:
                    box_data = predictions[i, j, :].tolist() + [i, j]
                    boxes.append(box_data)
        # NMS
        boxes_nms = []
        while boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            boxes_nms.append(boxes[0])
            boxes = [box for box in boxes if iou(boxes[0], box, img.size) < iou_threshold]
        # Borders
        draw = ImageDraw.Draw(img)
        for box in boxes_nms:
            # box: [p, x, y, w, h, i, j]
            size = img.size
            x_center = int((box[6] + box[1]) * (size[0] / 20))
            y_center = int((box[5] + box[2]) * (size[1] / 20))
            x_min = int(x_center - box[3] * size[0] / 20)
            x_max = int(x_center + box[3] * size[0] / 20)
            y_min = int(y_center - box[4] * size[1] / 20)
            y_max = int(y_center + box[4] * size[1] / 20)
            draw.line((x_min, y_min, x_min, y_max), '#00FF00', width=3)
            draw.line((x_min, y_min, x_max, y_min), '#00FF00', width=3)
            draw.line((x_max, y_min, x_max, y_max), '#00FF00', width=3)
            draw.line((x_min, y_max, x_max, y_max), '#00FF00', width=3)
    return img, predictions[:, :, 0]

def get_original_detected(img_path='./resources/tt100k_2021/train/23.jpg',
                          labels_json=None):
    if labels_json is None:
        labels_json = {}
    # Get id
    id = img_path.split('/')[-1][:-4]
    img = Image.open(img_path)
    # Get corresponding labels
    labels = torch.tensor(labels_json[str(id)]).view(20, 20, 5)
    draw = ImageDraw.Draw(img)
    # Draw correct boxes
    for i in range(20):
        for j in range(20):
            if labels[i][j][0] == 0:
                continue
            else:
                size = img.size
                x_center = (j + labels[i][j][1]) * (size[0] // 20)
                y_center = (i + labels[i][j][2]) * (size[1] // 20)
                x_min = x_center - labels[i][j][3] * size[0] // 2
                x_max = x_center + labels[i][j][3] * size[0] // 2
                y_min = y_center - labels[i][j][4] * size[1] // 2
                y_max = y_center + labels[i][j][4] * size[1] // 2
                draw.line((x_min, y_min, x_min, y_max), 'red', width=2)
                draw.line((x_min, y_min, x_max, y_min), 'red', width=2)
                draw.line((x_max, y_min, x_max, y_max), 'red', width=2)
                draw.line((x_min, y_max, x_max, y_max), 'red', width=2)
    return img

# Validation function for a whole model
def final_validation(model, device, data_loader, min_confidence=0.5):
    ious = []
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    # For each image
    with torch.no_grad():
        for images, labels in tqdm.tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device).view(-1, 20, 20, 5)
            outputs = model(images).view(-1, 20, 20, 5)
            # For each image in the batch, for each pixel calculate iou
            for b in range(labels.size(0)):
                for i in range(20):
                    for j in range(20):
                        prediction = outputs[b, i, j].tolist() + [i, j]
                        label = labels[b, i, j].tolist() + [i, j]
                        # Get relative boxes (full image)
                        # Count tp, fp, fn, tn according to confidence and save IoUs
                        predicted_sign = prediction[0] >= min_confidence
                        is_sign = label[0]
                        iou_value = iou(prediction, label, images[b].size())
                        if is_sign and predicted_sign:
                            total_tp += 1
                            ious.append(iou_value)
                        elif is_sign and not predicted_sign:
                            total_fn += 1
                        elif not is_sign and predicted_sign:
                            total_fp += 1
                        else:
                            total_tn += 1
    return {'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn,
            'ious': ious}