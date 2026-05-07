import json
from datasets import load_dataset

def main():
    print("⏳ Đang tải dataset medalpaca/medical_meadow_medqa...")
    # Tải dataset
    ds = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    
    # Shuffle dataset
    ds_shuffled = ds.shuffle(seed=42)
    
    # Lọc bỏ các mẫu có output quá ngắn (theo yêu cầu Lab: < 10 tokens, xấp xỉ 30-40 ký tự)
    # Chúng ta lọc trên toàn bộ tập trước
    ds_filtered = ds_shuffled.filter(lambda x: len(str(x['output']).split()) >= 10)
    
    # Lab yêu cầu từ 100 - 500 samples. Chọn đúng 500 samples
    print(f"✅ Số lượng thỏa mãn: {len(ds_filtered)} samples.")
    ds_subset = ds_filtered.select(range(500))

    # Chia 90% train, 10% eval
    split_ds = ds_subset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds['train']
    eval_ds = split_ds['test']
    
    print(f"📊 Train set: {len(train_ds)} samples")
    print(f"📊 Eval set: {len(eval_ds)} samples")
    
    # Lưu ra file JSON
    train_ds.to_json("train_dataset.json", force_ascii=False, indent=2)
    eval_ds.to_json("eval_dataset.json", force_ascii=False, indent=2)
    
    print("🎉 Hoàn tất! Đã lưu thành công `train_dataset.json` và `eval_dataset.json`.")

if __name__ == "__main__":
    main()
