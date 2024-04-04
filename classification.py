from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset
import numpy as np

def get_prob_vectors(dataset, feature_extractor, model, num_classes=1000):
    prob_vectors = torch.zeros((len(dataset), num_classes))
    for i, image in enumerate(dataset):
        image = image.convert('RGB')
        inputs = feature_extractor(image, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        prob_vectors[i] = torch.nn.functional.softmax(logits, dim=1)
        predicted_label = logits.argmax(-1).item()
        print(model.config.id2label[predicted_label])
    return prob_vectors

def compute_cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = torch.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -torch.sum(targets * torch.log(predictions + 1e-9))/N
    return ce

label_dataset = load_dataset('imagefolder', data_dir='results/inpainting/label')
recon_dataset = load_dataset('imagefolder', data_dir='results/inpainting/recon')
label_dataset = label_dataset['train']['image'][:len(recon_dataset['train']['image'])]
recon_dataset = recon_dataset['train']['image']

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')

label_prob_vectors = get_prob_vectors(label_dataset, feature_extractor, model)
recon_prob_vectors = get_prob_vectors(recon_dataset, feature_extractor, model)
label_prob_avg = torch.sum(label_prob_vectors, dim=0)/len(label_dataset)
recon_prob_avg = torch.sum(recon_prob_vectors, dim=0)/len(recon_dataset)

print('TV between true and reconstructed:', 0.5 * torch.sum(torch.absolute(label_prob_avg - recon_prob_avg)))

label_combined_class_avg = torch.zeros(2)
label_combined_class_avg[0] = torch.sum(label_prob_avg[:500])
label_combined_class_avg[1] = torch.sum(label_prob_avg[500:1000])

recon_combined_class_avg = torch.zeros(2)
recon_combined_class_avg[0] = torch.sum(recon_prob_avg[:500])
recon_combined_class_avg[1] = torch.sum(recon_prob_avg[500:1000])

print('TV between true and reconstructed (combined classes):', 0.5 * torch.sum(torch.absolute(label_combined_class_avg - recon_combined_class_avg)))


cross_entropy = compute_cross_entropy(recon_prob_vectors, label_prob_vectors)
print(cross_entropy)

cross_entropy = compute_cross_entropy(label_prob_vectors, label_prob_vectors)
print(cross_entropy)
#inputs = feature_extractor(image, return_tensors='pt')

#with torch.no_grad():
    #logits = model(**inputs).logits

#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])

