# 母婴闭域 RAG 技术方案

> 产品增加情绪支持双通道后的边界与路由，请以 [`产品边界与双通道方案-v2.md`](./产品边界与双通道方案-v2.md) 为准。本文件仍作为知识问答和 RAG 底座方案。

> 版本：v1.0 · 2026-09-15
>
> 目标：构建一个只依据已审核 Markdown 知识库回答的母婴健康科普助手。系统不做诊断、处方、用药剂量计算、疫苗禁忌判断或个体化治疗建议。

## 1. 结论先行

当前 Demo 的方向是正确的：先做知识范围和证据门禁，再调用 LLM。下一阶段不建议直接把开放式 Agent 加进来，而应强化四个独立边界：

1. **知识边界**：问题必须属于允许的主题，并且命中已发布文档。
2. **证据边界**：回答中的每个关键结论必须能对应到检索片段。
3. **安全边界**：急症、诊断、处方、药物剂量和疫苗禁忌走固定流程。
4. **运营边界**：知识文档必须审核、版本化、可撤回，模型调用和拒答必须留痕。

推荐演进路线：

```text
当前 Demo
  -> FastAPI + Pydantic + SQLite/PostgreSQL
  -> embedding + hybrid retrieval + reranker
  -> scope classifier + evidence grader
  -> 结构化回答 + 引用校验
  -> 审核发布、版本回滚、评测门禁、可观测性
```

## 2. GitHub 项目调研与取舍

以下项目来自官方 GitHub 仓库。星标和版本会变化，正式选型前应重新检查 release、依赖漏洞、许可证和维护活动。

| 项目 | 许可证 | 定位 | 建议 |
|---|---|---|---|
| [RAGFlow](https://github.com/infiniflow/ragflow) | Apache-2.0 | 全栈 RAG 引擎，文档解析、混合检索、引用和工作流 | 适合快速搭建内部知识库或作为解析/检索参考；部署较重 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | 代码优先的数据与检索框架 | 适合自研闭域管线，组件灵活；安全边界需要自己实现 |
| [Haystack](https://github.com/deepset-ai/haystack) | Apache-2.0 | 显式、可审计的 Pipeline 编排 | 适合生产检索流程和模型替换，推荐作为自研路线候选 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT | 有状态流程编排 | 适合实现 retrieve → grade → answer/refuse → human 的状态图，不必为 MVP 引入 Agent |
| [Milvus](https://github.com/milvus-io/milvus) | Apache-2.0 | 向量数据库和混合检索 | 数据量大、需要独立向量服务或多租户时选择 |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL License | PostgreSQL 向量扩展 | 中小规模首选，业务数据与向量可同库，运维成本低 |
| [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | MIT | BGE embedding/reranker 模型工具 | 中文场景可评估 `bge-m3` 与 `bge-reranker-v2-m3` |
| [MinerU](https://github.com/opendatalab/MinerU) | 项目自定义开源许可 | 中文复杂文档转 Markdown/JSON | 当前只收 Markdown，可作为后续 PDF 导入的解析器候选；需单独审许可证 |
| [Docling](https://github.com/docling-project/docling) | MIT | 多格式文档结构化解析 | 后续需要 DOCX/PDF/HTML 时作为通用候选 |
| [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Apache-2.0 | 输入、对话、检索、执行、输出 rails | 适合流程护栏和越狱检测，不是医学事实验证器 |
| [Ragas](https://github.com/vibrantlabsai/ragas) | Apache-2.0 | RAG 评测 | 评估检索和生成质量，不等于医疗正确性 |
| [DeepEval](https://github.com/confident-ai/deepeval) | Apache-2.0 | LLM 应用测试和 CI 门禁 | 适合把拒答、引用和忠实度加入回归测试 |
| [garak](https://github.com/NVIDIA/garak) | Apache-2.0 | LLM 红队扫描器 | 只用于上线前和定期攻击测试，不是运行时拦截器 |
| [MedSafetyBench](https://github.com/AI4LIFE-GROUP/med-safety-bench) | MIT | 医疗安全请求与安全回复数据集 | 可借鉴危险请求分类和拒答测试；必须补充母婴专项集 |

### 推荐组合

**MVP：** 当前 Python 服务 + Markdown + OpenAI-compatible LLM/Embedding API。

**生产自研：**

```text
FastAPI
+ PostgreSQL + pgvector
+ BGE-M3 或供应商 embedding API
+ BGE reranker-v2-m3 或商业 rerank API
+ LangGraph（只用于确定性状态流）
+ NeMo Guardrails（可选）
+ Langfuse / OpenTelemetry
+ Ragas + DeepEval + garak
```

**快速平台化：** RAGFlow 负责文档、知识库和引用，外置安全网关负责范围分类、急症分流、敏感主题拒答和输出校验。Dify/FastGPT 需要在多租户 SaaS 前进行许可证审查。

## 3. 生产级闭域流程

```text
POST /ask
  |
  v
输入校验与限流
  |
  v
PII 脱敏 + Prompt Injection 检测
  |
  v
确定性安全规则
  |-- 急症 -> 固定就医提示
  |-- 诊断/处方/剂量/禁忌 -> 固定拒答或人工升级
  |
  v
Scope Classifier
  |-- out_of_scope -> 拒答
  |-- in_scope -> 继续
  |
  v
Hybrid Retriever
  |-- metadata filter：主题、年龄、地区、版本、审核状态
  |-- dense retrieval
  |-- BM25/sparse retrieval
  |-- reranker
  |
  v
Evidence Gate
  |-- 无证据/证据冲突/低分 -> 拒答或人工复核
  |-- 足够证据 -> 继续
  |
  v
LLM Structured Generation
  |
  v
Claim/Citation Validator + Output Safety
  |-- 不通过 -> 固定拒答
  |-- 通过 -> 回答 + 引用 + 时间/版本
```

核心规则是：**“是否属于主题”与“是否有证据”是两个判断，不能只用一个相似度阈值代替。**

## 4. 如何发现“不应该回答”

### 4.1 四路信号

```json
{
  "scope": "in_scope",
  "topic": "infant_feeding",
  "risk_level": "L1",
  "emergency": false,
  "evidence_sufficient": true,
  "requires_human": false
}
```

判定器由四路组成：

1. **确定性规则**：急症词典、药品/剂量词典、诊断和处方词典。用于一票否决，不依赖 LLM。
2. **轻量分类器**：训练或微调中文意图分类模型，输出允许主题、风险等级和是否需要升级。
3. **检索证据**：使用 embedding + BM25 + reranker 判断问题是否被知识库覆盖。
4. **结构化 LLM 兜底**：只负责处理边界模糊的问题，输出必须符合 JSON Schema，不能直接决定急症放行。

### 4.2 建议的拒答决策

```text
命中急症规则                    -> emergency_response
命中药品剂量/处方/诊断/禁忌主题   -> hard_refusal
scope classifier 判定越界         -> out_of_scope
检索没有达到证据门槛              -> insufficient_evidence
证据互相冲突或已过期              -> conflict_or_stale
生成结果无法绑定引用              -> generation_failed
```

拒答原因应返回内部结构化码，前端展示用户友好的文案：

```text
out_of_scope：这个问题不在当前知识库服务范围内。
insufficient_evidence：知识库中没有足够可靠的信息支持回答。
hard_refusal：这个问题涉及个体化医疗判断，系统不能在线给出建议。
emergency：描述可能涉及紧急情况，请尽快联系急诊或拨打 120。
```

### 4.3 不应只相信相似度

向量相似度高不代表医学上正确。知识库中如果有错误、过期或被投毒的文档，RAG 仍可能给出高分。因此需要：

- 来源白名单和人工审核状态过滤
- 生效日期/失效日期过滤
- 文档版本和 embedding 模型版本绑定
- 重排后进行证据充分性检查
- 回答拆解为原子 claim，逐条绑定 chunk
- 证据不足时允许模型输出“不知道”

## 5. 知识库治理

### 5.1 Markdown 文档元数据

建议每个文档保存独立 manifest：

```json
{
  "document_id": "who-infant-feeding-2023",
  "title": "婴幼儿喂养基础知识",
  "source_url": "https://www.who.int/...",
  "publisher": "WHO",
  "published_at": "2023-10-16",
  "effective_from": "2023-10-16",
  "effective_to": null,
  "review_status": "approved",
  "reviewer": "nutrition-reviewer",
  "content_hash": "sha256:...",
  "embedding_model": "model-name@version",
  "schema_version": 1
}
```

### 5.2 发布流程

```text
导入 Markdown
  -> UTF-8/大小/扩展名校验
  -> PII 与 Prompt Injection 扫描
  -> 标题、来源、发布日期完整性检查
  -> 专业人员审核
  -> 生成 chunk 和 embedding
  -> 自动评测
  -> 发布为 approved 版本
```

未审核文档只能进入草稿索引，不能参与线上回答。删除或过期操作应生成审计事件，而不是直接物理覆盖。

## 6. LLM 与 Embedding API

当前 Demo 已支持 OpenAI-compatible 接口：

```text
LLM          POST /v1/chat/completions
Embedding    POST /v1/embeddings
```

生产建议：

- Key 只从 Secret Manager/Kubernetes Secret 注入，配置页面不应成为生产密钥管理入口。
- 配置页面仅适合本地 Demo，生产环境应关闭 `/api/config` 写接口。
- 记录模型名称、版本、请求 ID、耗时、token/cost，但不要记录明文 Key 和未经脱敏的健康信息。
- LLM 失败、超时或输出格式错误时，默认拒答，不用开放域模型兜底。
- Embedding 模型变更时创建新索引，完成离线评测后原子切换，不要混用不同向量空间。
- 生产环境为 LLM、Embedding 和 Reranker 设置独立超时、重试上限和熔断。

## 7. 安全工具的正确使用方式

### 运行时

- **NeMo Guardrails**：控制输入/输出和对话流程。
- **Guardrails AI**：做 JSON Schema、PII、格式和自定义验证器。
- **Prompt Guard/Llama Guard**：可作为注入或通用内容分类信号，需先验证中文效果和许可证。
- **业务规则引擎**：急症、药品、诊断和疫苗边界必须自建，优先级最高。

### 离线

- **garak**：测试越狱、注入、数据泄露和危险输出。
- **Ragas**：context precision/recall、faithfulness、answer relevancy。
- **DeepEval**：将拒答、引用、忠实度和回归测试接入 CI。
- **RAGChecker**：做 claim-level 检查时可作为研究型诊断工具。
- **MedSafetyBench**：构造医疗危险请求测试集的参考数据。

重要边界：这些工具都不能单独证明医疗安全。Ragas faithfulness 高，只能说明答案忠于检索上下文；不能证明上下文本身正确。

## 8. 评测体系

### 8.1 数据集组成

至少维护五类用例：

1. **域内可答**：母乳、辅食、新生儿日常护理、基础疫苗信息。
2. **域外问题**：奶粉品牌推荐、购物、成人健康、普通闲聊。
3. **高风险请求**：诊断、处方、药物剂量、停药、疫苗禁忌和补种计算。
4. **急症表达**：同一危险信号的正式、口语、错别字和多轮表达。
5. **攻击样本**：直接越狱、间接文档注入、编码/分隔符绕过、诱导披露系统提示。

### 8.2 指标

安全指标优先于平均回答质量：

```text
急症召回率                 >= 99%（上线门槛）
禁止主题自由生成率          = 0
危险回答违规率              = 0
引用存在且对应证据率        >= 99%
域外问题拒答准确率          由金标集确定
正常问题过度拒答率          <= 5%（需结合安全权衡）
```

质量指标：

```text
Context Precision / Recall
Faithfulness
Answer Relevancy
Claim-level citation accuracy
首 token 延迟、P95 总延迟、API 错误率
```

医疗正确性必须由儿科/妇产科/营养专业人员标注金标集，不能只依靠 LLM-as-a-judge。

## 9. 对当前 Demo 的优化优先级

### P0：上线前必须完成

- 将 `ThreadingHTTPServer` 替换为 FastAPI/Uvicorn，并加鉴权、HTTPS、限流和健康检查。
- `/api/config` 写接口仅在本地 Demo 开启，生产关闭并使用 Secret Manager。
- 增加独立 scope classifier，不再只用词法/向量阈值判断范围。
- 将急症词典、药品词典和诊断词典外置版本化，增加口语和错别字测试。
- LLM 输出改为 JSON Schema，校验 answer、citations、risk_level 和 refusal_reason。
- 知识库增加审核状态、来源、版本、有效期和 content hash。

### P1：显著提升回答质量

- 接入 BGE-M3 或 embedding API。
- 接入 BGE reranker-v2-m3 或合规的 rerank API。
- 混合检索使用加权融合或 RRF，不再简单使用 `max(词法分数, 向量分数)`。
- 增加 evidence sufficiency/claim verification 节点。
- 使用 pgvector 保存向量，SQLite 保存本地 Demo 元数据。
- 增加引用点击定位、文档版本展示和回答反馈。

### P2：运营和规模化

- Langfuse 或 OpenTelemetry 记录完整 trace。
- Ragas/DeepEval 接入 CI，garak 定期红队。
- 增加草稿、审核、发布、撤回、回滚后台。
- 分离静态文件、API、检索服务和模型调用服务。
- 需要复杂 PDF/扫描件时再引入 MinerU/Docling，不提前扩大输入面。

## 10. 最小生产 API 设计

```text
POST /v1/documents                上传 Markdown 草稿
POST /v1/documents/{id}/approve   审核发布
POST /v1/documents/{id}/revoke    撤回版本
GET  /v1/documents                文档与版本列表
POST /v1/ask                      闭域问答
POST /v1/evaluations/run          运行离线评测
GET  /health                      存活检查
GET  /ready                       依赖就绪检查
```

`/v1/ask` 建议返回：

```json
{
  "request_id": "req_123",
  "status": "answered",
  "answer": "...",
  "risk_level": "L0",
  "refusal_reason": null,
  "citations": [
    {
      "document_id": "who-infant-feeding-2023",
      "version": 1,
      "chunk_id": "chunk_01",
      "quote": "...",
      "score": 0.86
    }
  ],
  "knowledge_snapshot": "snapshot_2026_09_15"
}
```

## 11. 最终建议

当前项目继续保持“只收 Markdown”的输入面是合理的。先把 Markdown 知识治理、scope classifier、证据门禁、结构化输出和评测做扎实，再考虑 PDF、图片、Agent 和多轮个性化。

对于这个 Demo，最值得先做的三项优化是：

1. **真实的中文 embedding + reranker**，减少“看起来相关”的误命中。
2. **独立范围/风险分类器**，把“知识库没有覆盖”和“高风险不能回答”区分开。
3. **引用级输出校验**，回答每个结论必须能回指已审核 Markdown 的具体分块。

这三项比更换更大的 LLM 更能提升闭域母婴问答的可靠性。
