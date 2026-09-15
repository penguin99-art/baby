# 母婴知识助手 Demo

一个严格闭域的 LLM + RAG 演示：只有在导入文档中检索到足够证据时才回答，越界问题会拒答，疑似急症会返回固定就医提示。

## 启动

```bash
python3 app.py
```

打开 http://localhost:8000。当前只支持导入 `.md`，数据保存在 `data/knowledge.json`。

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
