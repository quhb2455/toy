from sklearn.metrics import accuracy_score, f1_score

def score(true_labels, model_preds, threshold=None) :
    model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    true_labels = true_labels.detach().cpu().numpy().tolist()
    return f1_score(true_labels, model_preds, average='macro')