# OpenCubee2_Web
# 🔁 Multi-Model Retrieval Server Setup (BEiT-3 + OpenCLIP + Gateway)

Hướng dẫn triển khai hệ thống truy hồi đa mô hình sử dụng Docker, BEiT-3, OpenCLIP và Gateway server.

## 📥 Bước 1: Tải File & Cấu Hình

- Tải **3 file cấu hình cần thiết** về thư mục dự án.  
- Chỉnh sửa lại đường dẫn trong các file nếu cần, đảm bảo trỏ đúng đến mô hình hoặc dữ liệu của bạn.

## 🚀 Bước 2: Chạy các dịch vụ trong Docker

Tất cả các bước sau đều thực hiện trong container Docker có tên:

`nguyenmv_aicity2025-nguyen_dfine_s`

---

**Terminal 1 — BEiT-3 Worker:**

docker exec -it nguyenmv_aicity2025-nguyen_dfine_s bash  
conda activate beit3  
export BEIT3_DEVICE=cuda:1   # 🧠 Chọn GPU để chạy BEiT-3  
uvicorn beit3_worker:app --host 0.0.0.0 --port 8001

---

**Terminal 2 — OpenCLIP Worker:**

docker exec -it nguyenmv_aicity2025-nguyen_dfine_s bash  
conda activate retrieval_env  
export OPENCLIP_DEVICE=cuda:7   # 🧠 Chọn GPU để chạy OpenCLIP  
uvicorn openclip_worker:app --host 0.0.0.0 --port 8002

---

**Terminal 3 — Gateway Server:**

docker exec -it fusion-search-server bash  
conda activate gateway_env  
uvicorn gateway_server:app --host 0.0.0.0 --port 18026

---

## 🔌 Bước 3: Kết nối từ xa qua SSH

ssh -L 18027:localhost:18027 nguyenmv@192.168.20.156  
**Password:** `ask owner`

---

## 🌐 Bước 4: Truy cập hệ thống

Trên trình duyệt, truy cập địa chỉ:

http://localhost:18027/

---

## 📝 Ghi chú

- Đảm bảo container Docker đang chạy đúng tên và đã cài đặt đầy đủ môi trường.  
- Kiểm tra các port (8001, 8002, 18026, 18027) không bị chặn bởi firewall hoặc xung đột bởi ứng dụng khác.  
- BEiT-3 và OpenCLIP cần chạy trên GPU tương thích (có thể tùy chỉnh `cuda:x` tùy server của bạn).
