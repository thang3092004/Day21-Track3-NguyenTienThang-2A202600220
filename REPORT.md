# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Tiến Thắng — 2A202600220
**Ngày nộp**: 07-05-2026
**Submission option**: B (GitHub + HF Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `medalpaca/medical_meadow_medqa`, 500 samples (450 train + 50 eval)
- **max_seq_length**: 512 (dựa trên phân tích p95 của tập dữ liệu)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: $0.18 (~30.4 phút @ $0.35/hr)
- **HF Hub link**: https://huggingface.co/thang3092004/qwen2.5-3b-medqa-lora-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 10.20 min  | 7.06 GB   | 1.365     | 3.917      |
| 16   | 3,686,400       | 10.23 min  | 4.98 GB   | 1.356     | 3.882      |
| 64   | 14,745,600      | 9.98 min   | 7.84 GB   | 1.350     | 3.857      |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png vào repository nếu có]
- **Quan sát**: Eval loss giảm đều đặn từ r=8 đến r=64 cho thấy không có dấu hiệu overfitting nghiêm trọng trên tập eval, mặc dù r=64 có số lượng parameters lớn gấp nhiều lần.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Q:A 16-year-old girl is brought to the emergency department unresponsive...
**Base**: A: There is an increase in the glycogen synthesis rate in this patient’s hepatocytes.
**Fine-tuned (r=16)**: E: The patient’s hypoglycemia inhibits glucagon release from pancreatic alpha cells.
**Nhận xét**: Cả hai đều chọn một đáp án trắc nghiệm ngắn gọn. Mô hình fine-tuned đã học được format output đúng với dữ liệu gốc thay vì trả lời dài dòng như một chatbot thông thường.

### Example 2
**Prompt**: Q:A 14-year-old girl presents to her pediatrician with complaints of repeated jerking of her neck for the past 2 years...
**Base**: A: History of repeated bouts of unprovoked obscene speech over the past year
**Fine-tuned (r=16)**: D: Decreased caudate volumes in magnetic resonance imaging (MRI) of the brain. ### Explanation: The presence of decreased caudate volumes...
**Nhận xét**: Mô hình fine-tuned không chỉ đưa ra đáp án đúng định dạng mà còn học được cách sinh ra phần giải thích (Explanation) rất sát với format của dataset y khoa.

### Example 3
**Prompt**: Q:A 31-year-old African American woman with a history of Addison's disease presents with widespread...
**Base**: A: The course usually is slowly progressive with spontaneous repigmentation in 15% of patients.
**Fine-tuned (r=16)**: C: Keloid formation is associated with regions of depigmentation. ### Explanation: Vitiligo is a chronic condition characterized by the loss...
**Nhận xét**: Tương tự câu 2, fine-tuned model tiếp tục chứng minh khả năng đưa ra đáp án kèm giải thích lâm sàng, vượt trội hơn so với base model.

### Example 4
**Prompt**: Q:A 35-year-old man presents with large tense blisters on the flexor surfaces of the upper extremities and trunk...
**Base**: B: Autoantibodies to desmoglein 3
**Fine-tuned (r=16)**: D: Linear band of immunoglobulin G (IgG) in the epidermal basement membrane
**Nhận xét**: Base model chọn sai sang bệnh Pemphigus vulgaris, trong khi Fine-tuned model chọn đúng cơ chế bệnh sinh của Bullous pemphigoid, chứng tỏ đã có sự cải thiện về kiến thức y khoa chuyên ngành.

### Example 5
**Prompt**: Q:A 67-year-old male with a past medical history of diabetes type II, obesity, and hyperlipidemia presents...
**Base**: C: Inhibition of cell wall synthesis
**Fine-tuned (r=16)**: A: Inhibition of the formation of the translation initiation complex
**Nhận xét**: Model fine-tuned đã bám sát tình huống (ngộ độc Aminoglycoside) và chọn đúng cơ chế ức chế quá trình dịch mã của vi khuẩn. Base model chọn sai hoàn toàn.

## 5. Conclusion về Rank Trade-off
Dựa trên kết quả thực nghiệm:
- **Rank có ROI tốt nhất**: r=16. Ở mức rank này, lượng tham số (3.6M) vừa đủ để mô hình bắt được cấu trúc cũng như kiến thức y khoa, trong khi lượng thời gian huấn luyện và VRAM tiêu thụ rất tối ưu (chỉ ~5GB VRAM). Eval loss và Perplexity (3.882) được cải thiện rõ rệt so với r=8.
- **Diminishing returns**: Xảy ra khi nâng từ r=16 lên r=64. Lượng parameters tăng gấp 4 lần (14.7M), Peak VRAM tăng lên gần 8GB nhưng Perplexity chỉ giảm rất ít (từ 3.882 xuống 3.857). Việc đánh đổi tài nguyên không mang lại giá trị tương xứng.
- **Recommendation**: Nếu triển khai production trên tập dữ liệu MedQA này, r=16 sẽ là lựa chọn cân bằng nhất. Nó cung cấp hiệu năng suy luận (Inference) và mức độ chính xác đáp ứng tốt cho miền y khoa mà không gây lãng phí tài nguyên máy chủ.

## 6. What I Learned
- **Về cấu trúc dữ liệu:** Học được cách chuẩn bị và đóng gói (format) một bộ dữ liệu Alpaca tiêu chuẩn phục vụ cho việc instruction fine-tuning, cụ thể là hiểu rõ tầm quan trọng của việc filter token length để tối ưu hóa `max_seq_length`.
- **Về kỹ thuật huấn luyện:** Trải nghiệm thực tế với QLoRA và Unsloth. Hiểu được cơ chế hoạt động của LoRA thông qua việc giới hạn số lượng tham số cần train (trainable parameters) giúp giảm thiểu đáng kể VRAM so với Full Fine-tuning.
- **Về việc đánh giá mô hình:** Hiểu được rằng để đánh giá một LLM, ta không thể chỉ dựa vào các con số định lượng (Eval Loss / Perplexity) mà còn phải tiến hành đánh giá định tính (Qualitative) thông qua việc sinh text trực tiếp (Inference) để xem mô hình có thực sự hiểu đúng format và format có bị vỡ hay không.
