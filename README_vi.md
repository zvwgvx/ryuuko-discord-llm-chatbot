# Ryuuko — Discord LLM Bot

Phiên bản: v5.0

Ryuuko là một Discord bot viết bằng Python, sử dụng một Large Language Model (LLM) — ví dụ OpenAI / Gemini / endpoint tương thích — để trả lời tin nhắn, xử lý lệnh và có tuỳ chọn lưu "memory" / cấu hình người dùng vào MongoDB.

---

## Tính năng chính

- Trả lời tin nhắn bằng LLM (prompt + system prompt theo user).
- Cấu hình người dùng (model, system prompt) với hai chế độ:
  - MongoDB (persistent) nếu bật `USE_MONGODB`.
  - File mode (fallback) lưu tại `config/user_config.json`.
- Hệ thống hàng đợi request bất đồng bộ (async) để tránh xử lý đồng thời nhiều request của cùng 1 user, kèm rate-limiting cơ bản.
- Wrapper gọi API LLM (module `call_api.py`).
- Các tiện ích xử lý văn bản (module `functions.py`) — ví dụ: chuyển LaTeX → ký tự Unicode, bảo vệ code blocks, tách message dài giữ nguyên bảng Markdown, v.v.
- Logging và xử lý lỗi thân thiện cho slash commands và legacy commands.

---

## Yêu cầu

- Python 3.10+ (3.11 khuyến nghị)  
- Pip để cài dependencies (xem `requirements.txt`)  
- Một Discord Bot token (Discord Developer Portal)  
- Một API key/endpoint cho LLM (OpenAI / Gemini / compatible)  
- (Tùy chọn) MongoDB nếu muốn lưu cấu hình/memory persistently  
- (Tùy chọn) Docker & docker-compose nếu muốn chạy MongoDB bằng container

---

## Cài đặt nhanh

```bash
# 1. Clone repository
git clone https://github.com/zvwgvx/Ryuuko.git
cd Ryuuko

# 2. Tạo virtual environment và cài phụ thuộc
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Chuẩn bị cấu hình (xem phần Cấu hình)
```

---

## Cấu hình (env / config.json)

Dự án đọc cấu hình từ `config.json`. Bạn có thể đặt giá trị trong file đó hoặc cung cấp tương đương. KHÔNG commit secrets (Discord token, API key, connection string).

Các biến quan trọng (trong `config.json` hoặc tương đương):
- DISCORD_TOKEN — token bot Discord (bắt buộc)  
- OPENAI_API_KEY — API key cho OpenAI (hoặc key tương ứng)  
- OPENAI_API_BASE — (tùy nếu dùng custom base URL)  
- OPENAI_MODEL — (mặc định có thể override per-user)  
- CLIENT_GEMINI_API_KEY, OWNER_GEMINI_API_KEY — (nếu tích hợp Gemini)  
- USE_MONGODB — true/false (bật MongoDB mode)  
- MONGODB_CONNECTION_STRING — URI MongoDB (nếu dùng)  
- MONGODB_DATABASE_NAME — tên database (mặc định: discord_openai_proxy)  
- REQUEST_TIMEOUT — (int) thời gian chờ gọi API  
- MAX_MSG — (int) giới hạn độ dài message xử lý

Ví dụ `.env` (KHÔNG lưu vào VCS):
```text
DISCORD_TOKEN=bot-token-here
OPENAI_API_KEY=sk-...
USE_MONGODB=true
MONGODB_CONNECTION_STRING=mongodb://user:pass@host:27017
```

---

## Cách chạy

```bash
# Chạy từ thư mục gốc
python ryuuko.py

# Hoặc chạy module chính
python src/main.py
```

Lưu ý:
- `ryuuko.py` thêm `src/` vào `sys.path` và gọi `main.bot.run(DISCORD_TOKEN)` — kiểm tra token hợp lệ trước khi chạy.
- Nếu dùng MongoDB, đảm bảo `MONGODB_CONNECTION_STRING` đúng và MongoDB reachable.

---

## Docker (tùy chọn)

Ví dụ Dockerfile tối giản (KHÔNG bake secrets):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "ryuuko.py"]
```

Docker-compose mẫu cho MongoDB:
```yaml
version: '3.8'
services:
  mongo:
    image: mongo:6
    restart: unless-stopped
    ports:
      - 27017:27017
    volumes:
      - mongo_data:/data/db
volumes:
  mongo_data:
```

---

## Cấu trúc thư mục & vai trò từng module

- ryuuko.py — entrypoint/launcher (thêm src vào sys.path, import main, start bot)  
- requirements.txt — phụ thuộc Python  
- config.json — mẫu/placeholder config (không chứa secrets thật)  
- src/  
  - main.py — định nghĩa bot Discord, event handlers, sync slash commands, xử lý shutdown, v.v.  
  - load_config.py — đọc `config.json`, khởi tạo biến cấu hình toàn cục, logger.  
  - call_api.py — wrapper gọi LLM (OpenAI/Gemini/custom).  
  - user_config.py — quản lý cấu hình per-user (model, system prompt); hỗ trợ MongoDB hoặc file fallback.  
  - memory_store.py — lưu memory tạm (in-memory) khi không dùng MongoDB; trimming theo giới hạn.  
  - mongodb_store.py — lưu trữ persistent: user configs, memories, supported models.  
  - request_queue.py — hàng đợi async (PriorityQueue) để xử lý request: tránh xử lý song song cho cùng 1 user, rate-limit, worker background.  
  - functions.py — utilities text processing, định nghĩa và đăng ký slash commands, on_message listener, process AI request callback.

---

## Lệnh / Commands (thực tế)

Danh sách các slash command chính (đăng ký trong `functions.setup()`):

- `/help` — Hiển thị danh sách lệnh và hướng dẫn ngắn. (everyone)  
- `/getid [mention_or_user]` — Hiển thị ID của bạn hoặc user được mention. (everyone)  
- `/ping` — kiểm tra phản hồi của bot / latency. (everyone)

Nhóm lệnh cấu hình (authorized users):
- `/set model <model>` — đặt model AI mặc định cho user.  
- `/set sys_prompt <prompt>` — đặt system prompt cho user.

Nhóm lệnh hiển thị (authorized users):
- `/show profile [user]` — hiển thị cấu hình của user.  
- `/show sys_prompt [user]` — xem system prompt hiện tại của user.  
- `/show models` — liệt kê các model được hỗ trợ.

Memory-related:
- `/memory` — thao tác liên quan tới memory per-user (xem chi tiết code).  
- `/clearmemory [user]` — xóa conversation history cho user (authorized).

Nhóm `add`, `remove`, `edit` — quản lý resources (models/profiles) với nhiều subcommands.  
Owner-only:
- `/auth <user>` — thêm user vào danh sách authorized (owner-only).  
- `/deauth <user>` — gỡ quyền authorized của user (owner-only).

Ghi chú: các subcommands và signature chi tiết nằm trong `src/functions.py`. Nếu cần, tôi có thể trích xuất và liệt kê đầy đủ subcommands.

---

## Flow hoạt động (tóm tắt)

1. Bot nhận message hoặc slash command từ Discord (main.py → functions).  
2. Kiểm tra quyền/authorized, lấy cấu hình user (user_config.py).  
3. Đưa request vào `request_queue` để đảm bảo tuần tự và rate-limit.  
4. Worker từ `request_queue` gọi `call_api.py` để gửi prompt tới LLM và nhận kết quả.  
5. Kết quả được xử lý (format/sanitize) bởi `functions.py` nếu cần, rồi gửi trả cho user.  
6. Nếu bật MongoDB, memory và cấu hình được lưu/đọc từ `mongodb_store.py`.

---

## Debug & Troubleshooting

- Bot không online: kiểm tra DISCORD_TOKEN, bot đã được invite và permissions.  
- Lỗi kết nối LLM: kiểm tra OPENAI_API_KEY / OPENAI_API_BASE / CLIENT_GEMINI_API_KEY.  
- Lỗi kết nối MongoDB: kiểm tra MONGODB_CONNECTION_STRING, mạng, firewall, credentials.  
- Rate-limit / request bị từ chối: kiểm tra thông báo lỗi trả về; bot có cơ chế ngăn user gửi nhiều request cùng lúc.  
- Kiểm tra logs console để xem trace và exception.

---

## Community & Support

Tham gia server Discord chính thức của dự án để thảo luận, hỏi đáp và nhận hỗ trợ:  
https://discord.gg/25mcSRMadU

---

## Bảo mật & vận hành

- KHÔNG commit secrets (token, keys, connection strings). Dùng `.gitignore` cho file config thật.  
- Nếu dùng Docker, truyền secrets qua env vars (docker run -e) hoặc secret manager.  
- Thêm index cho collection MongoDB có truy vấn thường xuyên (user_id, model names).  
- Thêm retry/backoff cho các thao tác network (LLM, DB).

---

## Đóng góp

1. Fork repository  
2. Tạo branch: feature/miêu-tả  
3. Mở PR với mô tả chi tiết  
4. Viết unit tests cho logic quan trọng (request_queue, user_config, mongodb_store)

---

## License

Mặc định: MIT.  
Dự án được cấp phép theo MIT License — xem tệp `LICENSE` để biết nội dung chi tiết.