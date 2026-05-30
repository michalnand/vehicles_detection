import json
import os
import matplotlib.pyplot as plt

def load_training_data(filepath):
    steps = []
    losses = []
    accuracies = []
    per_class_accuracies = []
    
    # Track extra training metrics
    macro_f1s = []
    macro_ious = []
    macro_precisions = []
    macro_recalls = []
    macro_specificities = []
    macro_balanced_accs = []
    macro_mccs = []
    
    imgs_per_sec = []
    data_prep_times = []
    forward_pass_times = []
    backward_pass_times = []
    
    num_classes = None

    if not os.path.exists(filepath):
        print(f"Warning: Training log file '{filepath}' not found. Skipping training plots.")
        return None

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            steps.append(data["step"])
            losses.append(data["loss"])
            accuracies.append(data["accuracy"])
            
            # Extract Detailed Hardware Pass Timing
            timing = data.get("timing", {})
            imgs_per_sec.append(timing.get("imgs/sec", 0))
            data_prep_times.append(timing.get("data_preparation", 0))
            forward_pass_times.append(timing.get("forward_pass", 0))
            backward_pass_times.append(timing.get("backward_pass", 0))
            
            # Extract Macro Robustness Metrics per step
            metric = data["metric"]
            num_classes = data["num_classes"]
            macro_f1s.append(metric.get("macro_f1_score", 0))
            macro_ious.append(metric.get("macro_iou", 0))
            macro_precisions.append(metric.get("macro_precision", 0))
            macro_recalls.append(metric.get("macro_recall", 0))
            macro_specificities.append(metric.get("macro_specificity", 0))
            macro_balanced_accs.append(metric.get("macro_balanced_accuracy", 0))
            macro_mccs.append(metric.get("macro_mcc", 0))
            
            tps = metric["tp_per_class"]
            tns = metric["tn_per_class"]
            fps = metric["fp_per_class"]
            fns = metric["fn_per_class"]
            
            step_per_class_acc = []
            for i in range(num_classes):
                total = tps[i] + tns[i] + fps[i] + fns[i]
                acc = (tps[i] + tns[i]) / total if total > 0 else 0.0
                step_per_class_acc.append(acc)
            
            per_class_accuracies.append(step_per_class_acc)
            
    return {
        "steps": steps,
        "losses": losses,
        "accuracies": accuracies,
        "per_class_accuracies": per_class_accuracies,
        "macro_f1s": macro_f1s,
        "macro_ious": macro_ious,
        "macro_precisions": macro_precisions,
        "macro_recalls": macro_recalls,
        "macro_specificities": macro_specificities,
        "macro_balanced_accs": macro_balanced_accs,
        "macro_mccs": macro_mccs,
        "imgs_per_sec": imgs_per_sec,
        "data_prep_times": data_prep_times,
        "forward_pass_times": forward_pass_times,
        "backward_pass_times": backward_pass_times,
        "num_classes": num_classes
    }

def generate_plots(train_data, test_data, results_path):
    if not train_data:
        return
    
    steps = train_data["steps"]
    
    # 1. Plot Training Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_data["accuracies"], color='b', label='Accuracy')
    plt.title('Training Accuracy vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + 'training_accuracy.png')
    plt.close()

    # 2. Plot Training Loss
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_data["losses"], color='r', label='Loss')
    plt.title('Training Loss vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + 'training_loss.png')
    plt.close()

    # 3. Plot Per-Class Accuracy
    plt.figure(figsize=(10, 5))
    num_classes = train_data["num_classes"]
    for i in range(num_classes):
        class_series = [step_acc[i] for step_acc in train_data["per_class_accuracies"]]
        plt.plot(steps, class_series, label=f'Class {i}')
    plt.title('Per-Class Training Accuracy vs. Steps')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_path + 'per_class_accuracy.png')
    plt.close()

    # 4. Macro Metrics Convergence
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_data["accuracies"], label='Global Accuracy', linestyle='--', color='gray', alpha=0.7)
    plt.plot(steps, train_data["macro_f1s"], label='Macro F1-Score', color='darkorange', linewidth=2)
    plt.plot(steps, train_data["macro_ious"], label='Macro IoU', color='teal', linewidth=2)
    plt.title('Macro Robustness Comparison During Training')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + 'macro_convergence.png')
    plt.close()

    # 5. Training Throughput and Overhead
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = 'tab:green'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Throughput (samples/sec)', color=color)
    ax1.plot(steps, train_data["imgs_per_sec"], color=color, label='Samples/Sec')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()   
    color = 'tab:purple'
    ax2.set_ylabel('Data Prep Time (sec)', color=color)
    ax2.plot(steps, train_data["data_prep_times"], color=color, linestyle=':', label='Data Prep Delay')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Hardware Efficiency and Data Loading Bottlenecks')
    fig.tight_layout()
    plt.savefig(results_path + 'hardware_efficiency.png')
    plt.close()

    # 6. Testing Imbalance vs F1-score
    if test_data:
        test_metrics = test_data.get("metric", {})
        classes = [f"Class {i}" for i in range(num_classes)]
        counts = test_metrics.get("class_counts", [])
        f1_scores = test_metrics.get("f1_score_per_class", [])

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        color = 'skyblue'
        ax1.set_xlabel('Class Category')
        ax1.set_ylabel('Test Support Volume', color='steelblue')
        ax1.bar(classes, counts, color=color, alpha=0.7, label='Sample Support')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        ax2 = ax1.twinx()
        color = 'crimson'
        ax2.set_ylabel('Test F1-Score', color=color)
        ax2.plot(classes, f1_scores, color=color, marker='o', linewidth=2, label='F1-Score')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.title('Impact of Class Imbalance on Model Success (Test Data)')
        fig.tight_layout()
        plt.savefig(results_path + 'test_imbalance_impact.png')
        plt.close()

def load_testing_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Testing log file '{filepath}' not found.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def build_markdown_report(train_data, test_data, output_path):
    md = []
    md.append("# Comprehensive Performance & Execution Report\n")
    
    # ==========================================
    # 1. TRAINING PERFORMANCE SECTION
    # ==========================================
    md.append("## 1. Training Progression Evaluation")
    if train_data:
        md.append(f"- **Total Training Steps Tracked**: {len(train_data['steps'])}")
        md.append(f"- **Final Training Loss**: {train_data['losses'][-1]:.6f}")
        md.append(f"- **Final Global Training Accuracy**: {train_data['accuracies'][-1]:.6f}")
        md.append(f"- **Final Macro F1-Score**: {train_data['macro_f1s'][-1]:.6f}")
        md.append(f"- **Final Macro IoU**: {train_data['macro_ious'][-1]:.6f}\n")
        
        md.append("\n")

        
        md.append("### Training Progress Visualizations")
        md.append("#### Global Loss & Accuracy History")
        md.append("![Global Loss](training_loss.png) ![Global Accuracy](training_accuracy.png)\n")
        md.append("#### Macro Metric Robustness & Hardware Profiles")
        md.append("![Macro Convergence](macro_convergence.png) ![Hardware Efficiency](hardware_efficiency.png)\n")
        md.append("#### Per-Class Base Line Accuracy Shifts")
        md.append("![Per-Class Accuracy](per_class_accuracy.png)\n")
    else:
        md.append("*No training logs found.*\n")
        
    # ==========================================
    # 2. TESTING RESULTS SECTION
    # ==========================================
    md.append("## 2. Final Test Evaluation Summary")
    if test_data:
        metrics_inner = test_data.get("metric", {})
        
        md.append("### Cross-Class Imbalance Risk Overview")
        md.append("![Test Imbalance Impact](test_imbalance_impact.png)\n")

        # 1. Global Metrics Table
        md.append("### Global Summary Test Metrics")
        md.append("| Global Performance Field | Log Value |")
        md.append("| :--- | :---: |")
        md.append(f"| **Total Test Samples Pool** | {test_data.get('num_samples')} |")
        md.append(f"| **Micro Accuracy Score** | {test_data.get('accuracy'):.5f} |")
        md.append(f"| **Global Intersection over Union (mIoU)** | {test_data.get('iou'):.5f} |")
        md.append(f"| **Global Dice/F1 Score** | {test_data.get('f1_score'):.5f} |")
        md.append(f"| **Macro Precision Average** | {metrics_inner.get('macro_precision'):.5f} |")
        md.append(f"| **Macro Recall Average** | {metrics_inner.get('macro_recall'):.5f} |")
        md.append(f"| **Macro Specificity Average** | {metrics_inner.get('macro_specificity'):.5f} |")
        md.append(f"| **Macro Balanced Accuracy Score** | {metrics_inner.get('macro_balanced_accuracy'):.5f} |")
        md.append(f"| **Macro Matthews Correlation Coefficient (MCC)** | {metrics_inner.get('macro_mcc'):.5f} |")
        md.append("\n")
        
        # New Section: Entropy & Certainty Spread (Requested Metrics Update)
        entropy_block = test_data.get("entropy", {})
        if entropy_block:
            md.append("### Prediction Entropy & Information Dynamics")
            md.append("| Certainty Classification Target Pool | Entropy Vector Score |")
            md.append("| :--- | :---: |")
            md.append(f"| **Global Mean Cross-Entropy** | {entropy_block.get('entropy'):.6f} |")
            md.append(f"| **Mean Entropy on Correct Predictions (High Certainty)** | {entropy_block.get('entropy_correct'):.6f} |")
            md.append(f"| **Mean Entropy on Incorrect Predictions (Confused/Wrong)** | {entropy_block.get('entropy_incorrect'):.6f} |")
            md.append("\n")

        # 2. Confusion Matrix Table
        md.append("### Confusion Matrix (Per-Class Breakdown)")
        md.append("| Class Identifier | True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) | Total Ground Support |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        num_classes = test_data.get("num_classes", 0)
        for i in range(num_classes):
            tp = metrics_inner["tp_per_class"][i]
            tn = metrics_inner["tn_per_class"][i]
            fp = metrics_inner["fp_per_class"][i]
            fn = metrics_inner["fn_per_class"][i]
            count = metrics_inner["class_counts"][i]
            md.append(f"| **Class {i}** | {tp} | {tn} | {fp} | {fn} | {count} |")
        md.append("\n")
        
        # 3. Per-Class Metrics Table
        md.append("### Comprehensive Per-Class Metric Arrays")
        md.append("| Class Identifier | Precision | Recall / Sensitivity | F1-Score / Dice | Isolated Class Accuracy |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for i in range(num_classes):
            precision = metrics_inner["precision_per_class"][i]
            recall = metrics_inner["recall_per_class"][i]
            f1 = metrics_inner["f1_score_per_class"][i]
            
            tp = metrics_inner["tp_per_class"][i]
            tn = metrics_inner["tn_per_class"][i]
            fp = metrics_inner["fp_per_class"][i]
            fn = metrics_inner["fn_per_class"][i]
            total = tp + tn + fp + fn
            class_acc = (tp + tn) / total if total > 0 else 0.0
            
            md.append(f"| **Class {i}** | {precision:.5f} | {recall:.5f} | {f1:.5f} | {class_acc:.5f} |")
        md.append("\n")
        
        # 4. Latency / Inference Timing Profiles
        timing = test_data.get("timing", {})
        if timing:
            md.append("### Test Inference Latency Profiles")
            md.append("| Statistical Distribution Slice | Execution Latency Target |")
            md.append("| :--- | :---: |")
            md.append(f"| **Mean Latency per Sample** | {timing.get('mean'):.6f} sec |")
            md.append(f"| **Standard Deviation ($\sigma$)** | {timing.get('std'):.6f} sec |")
            md.append(f"| **68th Percentile (P68)** | {timing.get('p68'):.6f} sec |")
            md.append(f"| **95th Percentile (P95)** | {timing.get('p95'):.6f} sec |")
            md.append(f"| **99.7th Percentile (P99.7)** | {timing.get('p997'):.6f} sec |")
            md.append(f"| **Maximum Outlier Peak Latency** | {timing.get('max'):.6f} sec |")
            md.append("\n")
            
    else:
        md.append("*No evaluation testing data found.*\n")
        
    with open(output_path, "w") as f:
        f.write("\n".join(md))
    print(f"Consolidated Markdown summary report generated at: '{output_path}'")

def main(results_path):
    print("Loading datasets...")
    train_data = load_training_data(results_path + "training.log")
    test_data = load_testing_data(results_path + "testing.log")
    
    print("Generating Matplotlib plots...")
    generate_plots(train_data, test_data, results_path)
    
    print("Compiling markdown structure...")
    build_markdown_report(train_data, test_data, results_path + "summary_report.md")

if __name__ == "__main__":
    results_path = "results/cnn_model_classifier/"
    main(results_path)