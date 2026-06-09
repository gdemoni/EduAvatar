# 师智分身 — 课程教学数字人

> 基于 RAG + LangGraph + Wav2Lip 的多模态 AI 智能助教数字人

---

## 项目简介

**师智分身**是一个课程教学数字人系统，将大语言模型（LLM）、检索增强生成（RAG）、语音合成（TTS）和数字人面部渲染技术融为一体，实现一个能讲课、会答疑、带表情的 AI 智能助教。

用户对着麦克风说话提问，系统自动识别语音，通过 LangGraph 智能 Agent 分析意图、检索课程知识库，生成口语化回答，最终由数字人形象配合唇形同步播报出来。

### 核心流程

```
语音 → ASR → 意图识别（讲解/解题）
    → FAISS 本地知识检索（未命中则网络搜索兜底）
    → 口语化内容改写 → TTS 语音合成
    → Wav2Lip 数字人口型同步 → WebRTC 实时推流
```

---

## 系统架构

```
┌─────────────────────────────────────────────┐
│                  师智分身                     │
├─────────────┬─────────────┬─────────────────┤
│   记忆       │    大脑      │      身体        │
│  RAG 知识库  │ LangGraph   │  Wav2Lip 数字人   │
│             │   Agent     │                 │
├─────────────┼─────────────┼─────────────────┤
│ FAISS 向量库 │ 10 个节点    │ 唇形同步渲染     │
│ BGE 嵌入模型 │ 6 个条件路由  │ WebRTC 实时推流  │
│ 多格式文档   │ 智能分流决策  │ EdgeTTS 语音合成 │
└─────────────┴─────────────┴─────────────────┘
```

| 层 | 核心技术 | 通俗说法 |
|------|---------|---------|
| 记忆 | RAG（FAISS + BGE） | AI 的"**记忆库**" |
| 大脑 | LangGraph Agent | AI 的"**大脑**" |
| 声带 | TTS（EdgeTTS） | AI 的"**声带**" |
| 嘴巴 | Wav2Lip | AI 的"**嘴巴**" |
| 传输 | WebRTC | AI 的"**高速公路**" |

---

## 技术栈

| 类别 | 技术选型 |
|------|---------|
| 后端框架 | LangGraph + LangChain |
| 大语言模型 | DeepSeek / 通义千问（OpenAI 兼容接口） |
| 向量数据库 | FAISS（本地 CPU 运行） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 语音合成 | EdgeTTS（免费） |
| 数字人渲染 | Wav2Lip / MuseTalk / Ultralight |
| 实时传输 | WebRTC |
| 文档解析 | PyPDF / python-docx |
| 网络搜索 | Tavily Search API |

---

## 功能特性

### 智能意图识别
自动判断用户是想**讲解知识点**还是**解答题目**，走不同的处理流程。

### 双链路处理

- **讲解链路**：本地检索 → 分层改写 → 口语化播报
- **解题链路**：复杂度判断 → 检索/直解 → 分步讲解 → 口语化播报

### 检索增强生成（RAG）
基于课程课件、教材构建本地知识库，回答有据可依，不凭空编造。本地未命中自动降级到网络搜索。

### 全链路语音适配
所有 AI 输出均为纯口语文本：禁止 Markdown 格式，数学公式自动转文字读法（如 x²→x的平方），短句逐行输出适配 TTS 停顿。

### 多模型可插拔
支持 Wav2Lip（快速入门）、MuseTalk（高画质）、Ultralight（低延迟）三种数字人渲染引擎，一键切换。

### 流式逐句播报
LLM 生成完整回答后按标点切句，首句延迟 < 2 秒即开始播报，无需等待全部文字生成。

---

## 应用场景

| 场景 | 说明 |
|------|------|
| 📚 课程辅导 | 学生课后随时提问，AI 助教基于课件/教材精准回答 |
| 🏫 翻转课堂 | 数字人代替老师完成基础知识讲授，课堂时间用于深度互动 |
| 🌐 远程教育 | 偏远地区学生也能享受 7×24 小时 AI 助教服务 |
| 🔒 数据安全 | 课程资料不出本地，全部基于本地 FAISS 向量数据库 |

---

## 快速开始

### 环境要求

- Python 3.11
- NVIDIA GPU 4GB+ 显存（可选，CPU 也可运行）
- CUDA 12.4（GPU 用户）

### 安装

```bash
# 创建虚拟环境
conda create -n szfs python=3.11
conda activate szfs

# 安装 PyTorch
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装项目依赖
pip install -r backend/requirements_backend.txt
pip install -r fronted/requirements_fronted.txt
```

### 构建知识库

```bash
cd rag
python rag_faiss_build.py --source_dir ../data --index_dir ../faiss_store
```

### 启动服务

```bash
# 终端 1：后端 Agent
cd backend/agent
pip install -e .
langgraph dev

# 终端 2：数字人前端
cd fronted/LiveTalking-main
python app.py --model wav2lip --avatar_id wav2lip256_avatar1
```

浏览器打开 `http://localhost:8010/dashboard.html` 即可使用。

---

## 开源致谢

本项目基于以下优秀的开源项目构建：

| 项目 | 用途 |
|------|------|
| [LiveTalking](https://github.com/lipku/LiveTalking) | 数字人前端框架 |
| [LangGraph ReAct Agent](https://github.com/langchain-ai/react-agent) | 后端 Agent 工作流引擎 |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM 应用开发框架 |
| [FAISS](https://github.com/facebookresearch/faiss) | 向量相似度检索 |
| [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | 唇形同步模型 |
| [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 中文嵌入模型 |

---

> 📌 **完整教程与源码**：[github.com/gdemoni/EduAvatar](https://github.com/gdemoni/EduAvatar)
