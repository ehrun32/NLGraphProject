import os
import numpy as np

prompt_styles = {
    "none": "ZERO-SHOT",
    "k-shot": "FEW-SHOT",
    "CoT": "CoT",
    "0-CoT": "0-CoT",
    "CoT+SC": "CoT+SC"
}


folders = {
    "none": {
        "easy": "gpt-4o-easy-none-20240409---18-34",
        "medium": "gpt-4o-medium-none-20240409---18-38",
        "hard": "gpt-4o-hard-none-20240409---18-42"
    },
    "k-shot": {
        "easy": "gpt-4o-easy-k-shot-20240409---19-01",
        "medium": "gpt-4o-medium-k-shot-20240409---19-06",
        "hard": "gpt-4o-hard-k-shot-20240409---19-11"
    },
    "CoT": {
        "easy": "gpt-4o-easy-CoT-20240409---19-20",
        "medium": "gpt-4o-medium-CoT-20240409---19-23",
        "hard": "gpt-4o-hard-CoT-20240409---19-27"
    },
    "0-CoT": {
        "easy": "gpt-4o-easy-0-CoT-20240409---19-31",
        "medium": "gpt-4o-medium-0-CoT-20240409---19-36",
        "hard": "gpt-4o-hard-0-CoT-20240409---19-40"
    },
    "CoT+SC": {
        "easy": "gpt-4o-easy-CoT-20240409---19-45+SC",
        "medium": "gpt-4o-medium-CoT-20240409---19-49+SC",
        "hard": "gpt-4o-hard-CoT-20240409---19-52+SC"
    }
}

log_root = "log/cycle"
difficulties = ["easy", "medium", "hard"]

# Compute accuracy per folder
def get_accuracy(folder):
    path = os.path.join(log_root, folder, "res.npy")
    res = np.load(path)
    return round(100 * res.sum() / len(res), 2)

# Print header
print(f"{'Method':<10} {'Easy':>7} {'Medium':>9} {'Hard':>7} {'Avg.':>7}")
print("-" * 45)

# Loop over styles and print formatted results
for style_key, label in prompt_styles.items():
    easy_acc = get_accuracy(folders[style_key]["easy"])
    med_acc = get_accuracy(folders[style_key]["medium"])
    hard_acc = get_accuracy(folders[style_key]["hard"])
    avg_acc = round((easy_acc + med_acc + hard_acc) / 3, 2)

    print(f"{label:<10} {easy_acc:7.2f} {med_acc:9.2f} {hard_acc:7.2f} {avg_acc:7.2f}")
