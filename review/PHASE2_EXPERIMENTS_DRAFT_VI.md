# Draft: thí nghiệm Phase II noisy ICL classification

Trạng thái: đề xuất để triển khai và chạy; chưa có kết quả thực nghiệm.
Nhánh: `phase2/rho-sweep-100k`. Code nền: `f766802`; bốn YAML rho sweep: `d4750ca`.
Tài liệu này bổ sung kế hoạch nghiên cứu, chưa tạo toàn bộ YAML hoặc sửa evaluator.

## 1. Câu hỏi nghiên cứu và phạm vi kết luận

Khảo sát khả năng đồng thời khớp nhãn nhiễu trong context và dự đoán tốt nhãn
sạch trên điểm mới, khi thay đổi tương quan đầu vào, mức nhiễu và độ dài context.

Trong thiết lập này, trọng số Transformer được meta-train qua nhiều task mới;
context là tập ví dụ của task tại inference. Vì vậy, ghi rõ đây là nghiên cứu
interpolation/generalization trong context, không đồng nhất với việc Transformer
interpolate một tập meta-training hữu hạn.

Self-query đưa lại một điểm đã có nhãn trong context. Error thấp ở phép đo này
có thể đến từ sao chép. Chỉ số đó riêng lẻ không đủ chứng minh benign overfitting.
Draft không giả định trước rằng tăng rho hoặc tăng context sẽ cải thiện clean risk.

## 2. Thiết lập chung

| Thành phần | Giá trị mặc định |
| --- | --- |
| Model | Garg tiny: embedding 64, 3 layer, 2 head |
| Dimension | d=20 |
| Context | m=40; sequence chứa m+1 điểm |
| Prior | w ~ N(0, I_d/d), bias=0 |
| Corruption | Independent label flip, eta=0.1 |
| Features | Gaussian stationary, Cov(x_t,x_(t+h)) = rho^h I_d |
| Optimizer | Adam, learning rate 0.0001; theo code hiện tại |
| Batch / thời lượng | 64 prompts/update, 100000 updates/run |
| Curriculum | Tắt: dims và points cố định, increment=0 |
| Training seeds | 0, 1, 2 khi làm bản lặp đầy đủ |
| Eval seed | 1000003, cố định giữa các training seeds |
| Eval budget | 1024 tasks, batch 16, query chunk 4 |
| Diagnostics | Self-query, LOO, exact-match copy; lưu per-task |
| Bayes | Tắt mặc định; excess risk chưa được tính |

Mỗi run có output và tên riêng chứa rho, eta, m và seed. Dùng config đã resolve
được lưu cùng checkpoint làm nguồn xác định thiết lập; không sửa config của run
đã train chỉ để đổi phân phối evaluation.

## 3. E1 — Độ ổn định của rho sweep qua training seeds

**Câu hỏi:** Xu hướng theo rho có lặp lại qua các lần meta-training không?

- Grid: rho thuộc {0, 0.3, 0.6, 0.9}, seeds thuộc {0, 1, 2}.
- Giữ eta=0.1, d=20, m=40 và 100000 updates.
- Tổng 12 runs. Bốn YAML hiện có mô tả seed 0, chưa có kết quả chạy.
- Khi tạo bản seed 1/2, đổi seed, out_dir và wandb.name; giữ eval seed cố định.

Đầu ra chính: clean query error, self-query error, corrupted-label memorization
accuracy theo rho. Báo từng seed, trung bình và độ lệch chuẩn giữa training seeds;
không gộp mọi task từ ba model thành một tập mẫu độc lập.

SE trên eval tasks và độ biến thiên giữa training seeds là hai nguồn bất định
khác nhau. Ba seeds chỉ cung cấp kiểm tra độ ổn định ban đầu.

## 4. E2 — Tương tác giữa rho và label noise

**Câu hỏi:** Khi nhiễu tăng, model giữ clean generalization hay chủ yếu khớp nhãn
nhiễu xuất hiện trong context? Sự đánh đổi có khác ở rho cao không?

- Grid: rho thuộc {0, 0.9}; eta thuộc {0.05, 0.1, 0.2, 0.4}; seeds {0,1,2}.
- Giữ d=20, m=40 và 100000 updates.
- Tổng 24 runs; tái sử dụng 6 runs eta=0.1 từ E1 nên cần thêm 18 runs.
- Dùng flip noise cho toàn grid này; không trộn probit vào cùng trục eta.

Đầu ra: clean/noisy query error, self-query error và memorization accuracy riêng
trên các context labels bị flip. Báo cả corrupted_count vì eta nhỏ có ít điểm
nhiễu hơn. So sánh noisy error với eta + (1-2*eta)*clean error như kiểm tra
nhất quán theo kỳ vọng; không yêu cầu residual bằng zero trên mẫu hữu hạn.

eta=0 hiện bị validator từ chối. Một đối chứng không nhiễu sẽ cần sửa validation,
logarithm/entropy và Bayes likelihood để xử lý biên, kèm test riêng trước khi chạy.

## 5. E3 — Query continuation so với query IID độc lập

**Câu hỏi:** Clean error thấp ở rho cao có phụ thuộc vào việc query là điểm tiếp
theo của chuỗi, gần các điểm context không?

Ưu tiên chạy trên checkpoint E1 trước khi mở rộng training grid.

Với cùng task weight, context và noisy context labels, tạo hai loại query:

1. Continuation: x_q = rho*x_m + sqrt(1-rho^2)*epsilon.
2. IID: x_q ~ N(0,I_d), độc lập với toàn bộ context, nhưng dùng cùng task weight.

Hai loại query giữ cùng marginal N(0,I_d). Không đổi rho của context đã sinh.
Sinh observed query labels theo cùng noise model; dự đoán query không được nhìn
thấy query label. Self-query và LOO tiếp tục dùng nguyên context labels ban đầu.

**Cần bổ sung evaluator:** một lựa chọn query_strategy; metadata/cache key ghi
chiến lược; paired per-task results; test bảo đảm context/nhãn không đổi giữa hai
nhánh. Không thể thực hiện đúng thí nghiệm này chỉ bằng YAML hiện tại.

Đầu ra chính: chênh lệch clean error IID trừ continuation, ghép cặp theo task.
Báo SE hoặc bootstrap theo task cho chênh lệch, rồi báo từng training seed.
Nếu gap lớn ở rho cao, đó là bằng chứng kết quả phụ thuộc query-context coupling;
chưa đủ để kết luận model chỉ làm nearest-neighbor hay không học quy luật.

Ở rho=0, hai protocol có cùng phân phối; đây là đối chứng phân phối quan trọng.
Nếu dùng chung Gaussian innovation tại rho=0, có thể kiểm tra equality trực tiếp.

## 6. E4 — Độ dài context quanh dimension

**Câu hỏi:** Tăng số ví dụ so với d có cải thiện clean generalization, và đi kèm
khả năng khớp nhãn nhiễu như thế nào?

- Grid: m thuộc {20,40,80}; rho thuộc {0,0.9}; seeds {0,1,2}.
- Giữ d=20, eta=0.1 và 100000 updates.
- Tổng 18 runs; tái sử dụng 6 runs m=40 từ E1 nên cần thêm 12 runs.
- Đặt đồng thời model.n_positions và curriculum.points.start/end bằng m+1;
  points.inc=0. Không chỉ thay riêng n_positions.

Đây là grid có cùng số optimizer updates, chưa có cùng chi phí: số điểm/prompt
tăng theo m và chi phí attention cũng thay đổi. Báo thời gian chạy và lượng dữ
liệu; nếu cần đối chứng cùng số điểm train, chạy thêm ngân sách updates điều chỉnh
theo m+1 và ghi rõ đó là một protocol khác.

Đồ thị đề xuất: clean query error và self-query error theo m/d; thêm corrupted
memorization và copy baseline. Không giả định m=d là interpolation threshold
của Transformer hoặc của bài classification này.

## 7. Metrics, artifacts và tiêu chí diễn giải

| Metric hiện có | Dùng để đánh giá |
| --- | --- |
| fresh_prompt_bce | Loss trên prompts mới, trung bình qua các prefix |
| memorization_error | Khớp observed context labels qua self-query |
| corrupted_memorization.accuracy | Khớp nhãn sai trên subset bị corruption |
| query_clean_error | Generalization tới clean label của query mới |
| query_noisy_error | Prediction error với observed query label |
| leave_one_out_error | Diagnostic khi bỏ cặp đang hỏi, target vẫn là observed label |
| copy_* | Đối chứng shortcut exact-match; unseen fallback hiện là +1 |
| clean_excess_risk_estimate | Mặc định null; chỉ có khi bật Bayes approximation và đạt ESS threshold |

Phân biệt SE theo task với biến thiên giữa training seeds. Các giá trị Bayes xấp
xỉ cần sensitivity theo số particles và seed, kiểm tra ESS; ESS cao riêng lẻ không
chứng nhận độ chính xác. Không điền clean error vào cột excess risk khi chưa có
baseline Bayes phù hợp.

Giữ config.yaml, state.pt, checkpoints định kỳ, training_metrics.jsonl và
phase2_metrics.json; thêm bảng tổng hợp có commit, checkpoint hash, seed, rho,
eta, m, query strategy và runtime. Evaluator mới cần version/cache key mới.

Chọn checkpoint cuối 100000 updates làm endpoint chính. Nếu khảo sát checkpoint
trung gian, xác định lịch trước và không chọn checkpoint bằng clean error trên
tập evaluation cuối. Lưu ý model_<i>.pt dùng chỉ số update zero-based trong code.

Một kết quả đáng theo đuổi là self-query error thấp đồng thời clean query error
thấp và ổn định trên query IID. Vẫn phải báo copy baseline, noise level và độ bất
định; chưa gọi đó là bằng chứng excess risk nhỏ nếu chưa tính baseline phù hợp.

## 8. Thứ tự thực hiện và ngân sách

1. Chạy GPU acceptance tests và phase2_smoke.yaml, kiểm tra checkpoint/evaluation.
2. Chạy E1: 12 runs; làm các biểu đồ rho ban đầu.
3. Triển khai E3 và đánh giá lại cùng checkpoint, không cần train thêm.
4. Chạy E2 nếu muốn khảo sát tương tác noise: thêm 18 runs.
5. Chạy E4 nếu muốn khảo sát context length: thêm 12 runs.

Toàn bộ E1+E2+E4 có 42 training runs phân biệt sau khi bỏ trùng, tổng 4.2 triệu
optimizer updates. E3 tăng chi phí evaluation. Chưa ước lượng giờ GPU vì chưa có
benchmark trên phần cứng đích; cấu hình khác m có thời gian/update khác nhau.

Draft chỉ mô tả các phép đo và giả thuyết để kiểm tra, chưa khẳng định kết quả.
