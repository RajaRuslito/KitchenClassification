import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, num_classes=2, device="cpu"):
    model = models.resnet50(weights=None)  # ga load pretrained lagi
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, class_names, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Gagal baca gambar {image_path}: {e}")
        return None, None, None

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    return class_names[pred.item()], probs[0][pred.item()].item(), pred.item()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_classification_report(report_dict, class_names):
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        values = [report_dict[label][metric] for label in class_names]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=class_names, y=values)
        plt.ylim(0, 1)
        plt.title(f"{metric.capitalize()} per Class")
        plt.ylabel(metric.capitalize())
        plt.show()

def main():
    # ==== Config ====
    model_path = "resnet50_kitchen.pth"
    class_names = ["clean", "dirty"]  # ganti sesuai dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Inference akan jalan di: {device}")

    # Load model
    model = load_model(model_path, num_classes=len(class_names), device=device)

    # ==== Tes 1 gambar ====
    test_image = "../dataset_dapur_split/val/dirty/000005.jpg"  # ganti path sesuai gambar lo
    pred, prob, _ = predict_image(model, test_image, class_names, device)
    if pred is not None:
        print(f"Gambar: {test_image} → Prediksi: {pred} ({prob:.2f})")

    # ==== Evaluasi semua data di folder test ====
    test_folder = "../dataset_dapur_split/val"
    print(f"\nEvaluasi folder: {test_folder}")

    y_true = []
    y_pred = []

    for root, _, files in os.walk(test_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, f)

                # ambil label asli dari nama folder
                true_label = os.path.basename(root)
                if true_label not in class_names:
                    continue
                true_idx = class_names.index(true_label)

                pred_label, prob, pred_idx = predict_image(model, img_path, class_names, device)
                if pred_label is not None:
                    y_true.append(true_idx)
                    y_pred.append(pred_idx)

    # ==== Hasil evaluasi ====
    print("\n=== Evaluasi Model ===")
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # ==== Plot grafik ====
    plot_confusion_matrix(cm, class_names)
    plot_classification_report(report, class_names)

if __name__ == "__main__":
    main()
