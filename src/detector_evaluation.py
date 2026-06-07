import os
import json
import numpy
import torch
from libs.dataset_magnetometer import *


if __name__ == "__main__":

    dataset_root_path = "/users/michal/datasets/car_detection_2/"
    #dataset_root_path = "/home/michal/datasets/car_detection_2/"
    results_path = "./results/rnn_model_detector/"
    os.makedirs(results_path, exist_ok=True)

    device = "cpu"
 
    dataset = DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_01-Lietavska_Lucka/02/")

    num_testing_samples = 1000

    # Detection parameters
    th = 0.2
    min_ratio_th = 0.02 # At least 2% of the window must be positive to count as an event

    true_positive  = 0
    true_negative  = 0
    false_negative = 0
    false_positive = 0  

    model = torch.load(results_path + "model_9.pt", weights_only=False, map_location='cpu')
    model.eval()

    for n in range(num_testing_samples):

        idx = numpy.random.randint(0, len(dataset))
        x, y_gt = dataset[idx] 

        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x_t)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.squeeze(0).detach().cpu().numpy()

        y_gt   = y_gt[:, 0]
        y_pred = y_pred[:, 0]
        
        # Calculate what percentage of the window is above the threshold
        gt_ratio = numpy.mean(y_gt > th)
        pred_ratio = numpy.mean(y_pred > th)    

        # Event is present in time window (Outlier-resistant)
        if gt_ratio >= min_ratio_th:
            # detector positive answer
            if pred_ratio >= min_ratio_th:
                true_positive += 1
            # detector negative answer
            else:
                false_negative += 1
        # event is not present in time window
        else:
            # detector find non existing event
            if pred_ratio >= min_ratio_th:
                false_positive += 1
            else:
                true_negative += 1

        if (n % 100) == 0:
            print(f"eval done {round(100 * n / num_testing_samples, 2)} %")

    total_samples = true_positive + true_negative + false_negative + false_positive

    # Calculate full evaluation metrics safely
    accuracy = (true_positive + true_negative) / total_samples if total_samples > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Structure final results into JSON format
    metrics_json = {
        "dataset_info": {
            "dataset_path": dataset_root_path,
            "total_samples_evaluated": total_samples
        },
        "hyperparameters": {
            "detection_threshold": th,
            "min_window_ratio_threshold": min_ratio_th
        },
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1_score, 4)
        }
    }

    # Print to console as pretty JSON string
    print("\n--- Evaluation Metrics Report ---")
    print(json.dumps(metrics_json, indent=4))

    # Save metrics JSON file
    output_file = os.path.join(results_path, "evaluation_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"\nMetrics successfully saved to: {output_file}")