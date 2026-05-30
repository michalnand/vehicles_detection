import json
import os
import matplotlib.pyplot as plt

def load_training_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Training log file '{filepath}' not found. Skipping training plots.")
        return None

    steps = []
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    ious = []
    mccs = []
    imgs_per_sec = []

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            steps.append(data["step"])
            losses.append(data["loss"])
            accuracies.append(data["accuracy"])
            ious.append(data["iou"])
            f1_scores.append(data["f1_score"])
            
            # Extract internal binary metric block elements
            metric = data.get("metric", {})
            precisions.append(metric.get("precision", 0))
            recalls.append(metric.get("recall", 0))
            mccs.append(metric.get("mcc", 0))
            
            timing = data.get("timing", {})
            imgs_per_sec.append(timing.get("imgs/sec", 0))
            
    return {
        "steps": steps, "losses": losses, "accuracies": accuracies,
        "precisions": precisions, "recalls": recalls, "f1_scores": f1_scores,
        "ious": ious, "mccs": mccs, "imgs_per_sec": imgs_per_sec
    }

def load_testing_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Testing log file '{filepath}' not found.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def generate_plots(train_data, test_data, results_path):
    if not train_data:
        return
    
    steps = train_data["steps"]
    
    # 1. Plot Training Loss & Accuracy (Dual Panel Subplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, train_data["losses"], color='crimson', linewidth=2)
    ax1.set_title('Binary Training Loss Progression')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(steps, train_data["accuracies"], color='royalblue', linewidth=2)
    ax2.set_title('Global Accuracy Progression')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_path + 'training_loss_accuracy.png')
    plt.close()

    # 2. Precision vs. Recall Convergence Trade-off
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_data["precisions"], label='Precision (Positive Predictive Value)', color='darkorange', linewidth=2)
    plt.plot(steps, train_data["recalls"], label='Recall (Sensitivity)', color='teal', linewidth=2)
    plt.plot(steps, train_data["f1_scores"], label='F1-Score', color='purple', linestyle='--', alpha=0.7)
    plt.title('Detection Harmony Balance (Precision vs. Recall)')
    plt.xlabel('Steps')
    plt.ylabel('Score Metric Metric Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(results_path + 'precision_recall_balance.png')
    plt.close()

    # NEW PLOT 3: Test Prediction Confidence Score Density Spline
    if test_data and "distribution" in test_data:
        dist = test_data["distribution"]
        hist_vals = dist.get("hist", [])
        bin_edges = dist.get("bin_edges", [])
        
        if hist_vals and bin_edges:
            plt.figure(figsize=(10, 5))
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            
            # Normalize counts to reflect a density probability distribution mapping
            total_counts = sum(hist_vals)
            density = [v / total_counts for v in hist_vals] if total_counts > 0 else hist_vals
            
            plt.bar(bin_centers, density, width=(bin_edges[1]-bin_edges[0])*0.8, color='cadetblue', alpha=0.6, label='Model Confidence Frequency')
            plt.axvline(x=test_data["metric"].get("threshold", 0.5), color='red', linestyle='--', label='Decision Boundary Threshold')
            
            plt.title('Model Confidence Distribution Profile (Test Data)')
            plt.xlabel('Predicted Probability Score')
            plt.ylabel('Relative Frequency Concentration Density')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_path + 'test_confidence_distribution.png')
            plt.close()

def build_markdown_report(train_data, test_data, output_path):
    md = []
    md.append("# Comprehensive Binary Detection Summary Performance Report\n")
    
    # ==========================================
    # 1. TRAINING EVALUATION METRICS
    # ==========================================
    md.append("## 1. Production Training Analysis")
    if train_data:
        md.append(f"- **Total System Evaluation Steps**: {train_data['steps'][-1]}")
        md.append(f"- **Final Optimization Step Loss Value**: {train_data['losses'][-1]:.6f}")
        md.append(f"- **Final Operational Tracking Accuracy**: {train_data['accuracies'][-1]:.5f}")
        md.append(f"- **Final Detection Core F1-Score**: {train_data['f1_scores'][-1]:.5f}")
        md.append(f"- **Final Intersection over Union (IoU)**: {train_data['ious'][-1]:.5f}\n")
        
        md.append("### Training Progress Graphics")
        md.append("#### Loss & Basic Accuracy Curves")
        md.append("![Loss & Accuracy](training_loss_accuracy.png)\n")
        md.append("#### Detection Harmony Trade-off Curves")
        md.append("![Precision Recall Balance](precision_recall_balance.png)\n")
    else:
        md.append("*No training trajectory metrics found data logged.*\n")

    # ==========================================
    # 2. TESTING PERFORMANCE METRICS
    # ==========================================
    md.append("## 2. Definitive Test Matrix Evaluation Evaluation")
    if test_data:
        metrics = test_data.get("metric", {})
        timing = test_data.get("timing", {})
        
        # Core Global Markdown Performance Metrics Table Matrix
        md.append("### Definitive Test Benchmark Outputs")
        md.append("| Target Benchmark Domain Metric Field | Evaluated Metric Output Score |")
        md.append("| :--- | :---: |")
        md.append(f"| **Total Processed Test Instance Volumes** | {test_data.get('num_samples'):,} |")
        md.append(f"| **Global Boundary Binary Classification Accuracy** | {test_data.get('accuracy'):.5f} |")
        md.append(f"| **Dice / F1 Evaluation Target Score Vector** | {test_data.get('f1_score'):.5f} |")
        md.append(f"| **Intersection over Union (IoU) Detection Index** | {test_data.get('iou'):.5f} |")
        md.append(f"| **Precision Index Performance** | {metrics.get('precision'):.5f} |")
        md.append(f"| **Recall / True Positive Sensitivity Rate** | {metrics.get('recall'):.5f} |")
        md.append(f"| **Specificity Index Value** | {metrics.get('specificity'):.5f} |")
        md.append(f"| **Balanced Accuracy Score Equation Average** | {metrics.get('balanced_accuracy'):.5f} |")
        md.append(f"| **Matthews Correlation Coefficient (MCC)** | {metrics.get('mcc'):.5f} |")
        md.append("\n")

        # Confidence Visualizer Graph Link
        if "distribution" in test_data:
            md.append("### Prediction Calibration & Score Distribution View Profiles")
            md.append("![Model Confidence Distribution Profile](test_confidence_distribution.png)\n")

        # Pure Confusion Matrix Matrix Array Setup
        md.append("### Binary Confusion Matrix Counts Structure")
        md.append("| True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) | Decision Boundary |")
        md.append("| :---: | :---: | :---: | :---: | :---: |")
        md.append(
            f"| {metrics.get('tp'):,} | {metrics.get('tn'):,} | "
            f"{metrics.get('fp'):,} | {metrics.get('fn'):,} | "
            f"Threshold @ {metrics.get('threshold', 0.5)} |"
        )
        md.append("\n")

        # Timing profiles
        if timing:
            md.append("### Batch Realtime Latency Execution Benchmarks")
            md.append("| Operational Percentile Slice | Sample Delay Profile Duration |")
            md.append("| :--- | :---: |")
            md.append(f"| **Average Inference Mean Delay** | {timing.get('mean'):.6f} sec |")
            md.append(f"| **Standard Deviation Profile Distribution Variance** | {timing.get('std'):.6f} sec |")
            md.append(f"| **95th Percentile Execution Window Limit (P95)** | {timing.get('p95'):.6f} sec |")
            md.append(f"| **Worst-Case System Latency Peak Outlier (Max)** | {timing.get('max'):.6f} sec |")
            md.append("\n")
    else:
        md.append("*No definitive evaluation benchmark dataset found records logged.*\n")

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    print(f"Binary classification markdown analysis file written cleanly to: '{output_path}'")

def main(results_path):
    print("Beginning execution pipeline setup for binary data...")
    train_data = load_training_data(results_path + "training.log")
    test_data = load_testing_data(results_path + "testing.log")
    
    print("Compiling tailored visualization output matrices...")
    generate_plots(train_data, test_data, results_path)
    
    print("Writing structural execution files report format summaries...")
    build_markdown_report(train_data, test_data, results_path + "summary_report.md")

if __name__ == "__main__":
    results_path = "results/rnn_model_detector/"
    main(results_path)