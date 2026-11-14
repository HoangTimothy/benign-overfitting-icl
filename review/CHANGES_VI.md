# Bản ghi nâng cấp để duyệt từng dòng

Ngày chuẩn bị: 2026-11-06. Upstream: https://github.com/dtsip/in-context-learning

Mốc gốc bất biến: `garg-2022-baseline` = `338337192195827c58356f9b627e5f999ea703d2`.
Repo được clone trực tiếp từ upstream, không lấy code hoặc dữ liệu nghiên cứu trong repo cũ bên cạnh.
Giữ lịch sử upstream và giấy phép MIT. Chưa bổ sung giả thuyết, noise model, thước đo hay thí nghiệm Benign overfitting.

## Các nhóm thay đổi

| Tệp | Thay đổi và lý do | Ảnh hưởng cần xác nhận |
| --- | --- | --- |
| `pyproject.toml`, `.python-version`, `uv.lock`, `environment.yml` | Python 3.12; dependency hiện đại; khóa cả dependency gián tiếp. Conda chỉ bootstrap Python/uv. | Thư viện mới có thể đổi kết quả số và mặc định baseline. Không khẳng định tái lập bitwise kết quả 2022. |
| `pyproject.toml` / Quinine overrides | Giữ parser và YAML inheritance cũ, nâng PyYAML, Munch, Funcy, Cytoolz, Cerberus, Gin, Toposort thay các pin cũ của Quinine 0.3.0. | Override metadata có chủ đích; chỉ dùng `uv sync --locked`, không dùng `pip install .`. Test parse các cấu hình task gốc. |
| `src/runtime.py`, `src/models.py`, `src/train.py`, `src/eval.py` | Tự chọn CPU/CUDA, bỏ lời gọi `.cuda()` bắt buộc; eval lấy device thật của model. | Cho phép chạy trên máy không GPU. GD baseline cũng hỗ trợ CPU. |
| `src/models.py` | Chọn GPT-2 attention `eager` tường minh. | Hạn chế đổi thuật toán attention mặc định sang SDPA; vẫn cần kiểm chứng sai số với stack 2022/checkpoint thật. |
| `src/train.py`, `src/eval.py` | `torch.load(..., weights_only=True)` và `map_location` rõ ràng. | Nạp state dict/tensor an toàn theo API hiện tại; checkpoint chứa object tùy biến không thuộc phạm vi hỗ trợ. |
| `src/train.py` / resume | Bắt đầu ở `train_step + 1`, thay vì lặp lại bước đã lưu. | Sửa hành vi upstream, ảnh hưởng trajectory của run resume. Chưa checkpoint RNG; chưa bảo đảm resume tương đương run liên tục. |
| `src/schema.py`, `src/train.py` | W&B mặc định disabled; thêm mode offline/online; thêm `device` và `eval_after_train`. | Config khoa học cũ giữ nguyên; logging online cần bật rõ ràng. Full eval vẫn mặc định bật. |
| `src/plot_utils.py` | Truyền keyword cho `seaborn.set_theme`. | Tương thích API keyword-only hiện tại. |
| `src/conf/smoke.yaml`, `tests/`, `.github/workflows/compatibility.yml` | Smoke training, roundtrip checkpoint, resume, causal masking, task/baseline, plotting; CI Windows/Linux CPU. | Smoke không đo convergence hay tái lập kết quả paper. CI được cấu hình, chỉ được coi đã pass khi có run thực tế. |
| `scripts/export_review.py`, `.gitattributes`, `.gitignore`, `README.md`, `review/` | Diff từng dòng, chuẩn hóa EOL và tài liệu. | File review sinh ra chỉ phản ánh commit HEAD; commit thay đổi trước khi xuất lại. |

## Các phần khoa học giữ nguyên

`tasks.py`, `samplers.py`, `curriculum.py`, `base_models.py`, notebook và các YAML thí nghiệm gốc được giữ nguyên nội dung. Cấu trúc GPT-2, dropout bằng 0, ghép token x/y, loss, optimizer Adam và curriculum gốc được giữ. Baseline sklearn/XGBoost dùng phiên bản mới nên mặc định nội bộ có thể khác 2022.

Không sửa các lỗi upstream khác ngoài phần đã liệt kê. Đặc biệt: nhánh seeded DecisionTree cần audit riêng nếu dùng; checkpoint chưa lưu RNG; lịch lưu checkpoint gốc có thể không lưu bước cuối nếu bước cuối không trùng chu kỳ. Đây là giới hạn cần xử lý có chủ đích trước thí nghiệm dài.

## Checklist xác nhận

- [ ] Duyệt dependency và Quinine overrides.
- [ ] Duyệt các dòng sửa trong `src/` bằng HTML hoặc `git diff`.
- [ ] Chấp nhận thay đổi resume và W&B mặc định.
- [ ] Chạy test/smoke trên GPU server mục tiêu và ghi CUDA/wheel/driver.
- [ ] Nạp checkpoint chính thức 2022 và đối chiếu metric trước khi tuyên bố tái lập paper.
- [ ] Sau xác nhận mới bắt đầu commit Benign overfitting.

## Nguồn API

- https://docs.pytorch.org/docs/stable/notes/serialization.html
- https://huggingface.co/docs/transformers/model_doc/gpt2
- https://docs.astral.sh/uv/concepts/projects/sync/

## Kết quả kiểm tra cục bộ

- Windows, Python 3.12.10, PyTorch 2.15.0+cpu, Transformers 5.22.1.
- `uv sync --locked`: cài thành công môi trường riêng `.venv`.
- `uv run --locked pytest -q` từ root: **17 passed**, 15.43 giây ở lượt cuối.
- `uv run --locked python train.py --config conf/smoke.yaml` từ `src`: hoàn thành 3 bước; checkpoint được tạo.
- `git diff --check`: pass.
- Các tệp khoa học được liệt kê là giữ nguyên đã được đối chiếu bằng Git với baseline.
- Chưa kiểm tra GPU, convergence dài hạn, notebook từ đầu đến cuối hay checkpoint chính thức 2022. Trạng thái CI xem trực tiếp trên PR.
