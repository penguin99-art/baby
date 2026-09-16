# 母婴知识助手 Demo

一个情绪陪伴优先的母婴顾问 LLM + RAG 演示：所有普通消息由同一个陪伴角色自然回应；识别到母婴知识需求时，会检索已导入 Markdown 并补充可追溯依据；疑似急症或心理危机仍返回固定安全提示。

## 启动

```bash
python3 app.py
```

打开 http://localhost:8000。当前只支持导入 `.md`，数据保存在 `data/knowledge.json`。

页面支持浏览器原生中文语音输入和回答朗读。语音识别的可用性及语音数据处理方式取决于浏览器实现；应用后端只接收识别后的文字，不接收或保存录音文件。

语音转写后需确认发送，自动朗读默认关闭；关闭自动朗读后仍可手动播放单条回复。录音取消保留已有草稿。

## 会话与验证

浏览器通过 HttpOnly / SameSite Cookie 访问本地 SQLite 会话。刷新页面或重启同一个服务后可恢复历史；会话在创建后 24 小时过期，下次请求时清理。页面的“删除会话”立即删除该会话历史，不删除知识库。数据文件 `data/conversations.sqlite3` 被 Git 忽略，但本地明文存储，不适合直接公开部署。

启用模型 API 后，最近对话会发送给供应商。当前没有账号系统、跨设备同步或长期健康画像。请只运行一个服务进程；SQLite 的会话串行锁不是多进程锁。`localhost` 与 `127.0.0.1` 是不同 Cookie 来源，请固定使用一种地址。

每条回复区分“模型生成 / 本地降级 / 固定能力”。`execution` 记录 planner 和 generator 的调用结果，包含未配置、超时、上游失败和输出阻断，不记录密钥或原始上游错误。

```bash
python3 -m unittest -v test_app test_conversation_store test_session_api
node --check static/app.js
```

API 变更：先 `GET /api/session` 获得 Cookie 与 revision，再 `POST /api/ask` 提交 `query`、`request_id`、`revision`；不再接受客户端 `history`。所有 POST 必须带 `X-Requested-With: BabyAssistant`。重复 ID 与相同内容返回缓存，不同内容或旧 revision 返回 409。`POST /api/session/clear` 删除会话。当前仅支持实际服务端口的 localhost / 127.0.0.1，限制跨站请求；这些限制不是生产鉴权的替代。

本轮实施记录与尚未完成的能力见 [M0/M1 实施记录](docs/M0-M1-实施记录.md)。目标架构见 [v4](docs/宝妈助手目标技术架构-v4.md)，现状对照与前沿迭代见 [v5](docs/现状对照与前沿迭代方案-v5.md)。

页面右上角的“设置”可以临时配置 LLM 和 Embedding API，并测试连通性。网页配置只写入当前 Python 进程内存，不会保存 API Key；服务重启后请使用环境变量或重新在页面填写。生产环境建议接入 Secret Manager，并通过 HTTPS 和鉴权保护配置接口。

## API 配置

配置文件示例见 `.env.example`。当前支持 OpenAI-compatible API：LLM 使用 `/v1/chat/completions`，Embedding 使用 `/v1/embeddings`。环境变量只在服务端读取，不能放进前端代码。

默认使用抽取式回答和轻量词法检索，方便离线演示。配置 API 后，导入时会为 Markdown 分块生成 embedding，提问时使用向量与词法结果的较高分进行检索，并将最多 3 个证据片段交给 LLM：

```bash
LLM_BASE_URL=https://api.example.com \
LLM_API_KEY=your-key \
LLM_MODEL=your-model \
EMBEDDING_MODEL=your-embedding-model \
python3 app.py
```

Embedding 默认复用 `LLM_BASE_URL` 和 `LLM_API_KEY`；如果是不同服务，单独配置 `EMBEDDING_BASE_URL` 和 `EMBEDDING_API_KEY`。更换 embedding 模型后应清空并重新导入文档，避免向量维度或语义空间不一致。

模型输出仍会经过基本的敏感内容拦截；生产环境应使用 Secret Manager、HTTPS、独立分类器、权威来源审核、输出校验、限流和人工复核。本 Demo 的 `data/knowledge.json` 是本地开发存储，不适合直接作为生产数据库。

## 参考

- `ref/ragflow`：RAGFlow 的文档解析、分块和可追溯引用思路
- [RAGFlow](https://github.com/infiniflow/ragflow)
- [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [Ragas](https://github.com/vibrantlabsai/ragas)

## Demo 边界

本项目仅用于展示闭域知识库问答，不提供诊断、处方、药品剂量或个体化治疗建议。知识库内容必须由专业人员审核后再用于真实用户。
