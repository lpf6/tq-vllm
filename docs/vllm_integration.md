# vLLM 自动集成机制

本文档解释 turboquant-vllm 如何自动集成到 vLLM 中。

## 自动集成原理

### 1. Entry Points 机制

Python 的 `entry_points` 允许包在安装时注册可被其他包发现的钩子。vLLM 使用这个机制来发现第三方插件。

在 [pyproject.toml](../pyproject.toml) 中配置：

```toml
[project.entry-points."vllm.general_plugins"]
tq4_backend = "turboquant_vllm.vllm:register_tq4_backend"
tq4_triton_backend = "turboquant_vllm.vllm:register_tq4_triton_backend"
tq4_flashinfer_backend = "turboquant_vllm.vllm:register_tq4_flashinfer_backend"
```

### 2. vLLM 的插件发现流程

当 vLLM 启动时，它会执行以下步骤：

1. **扫描 Entry Points**：vLLM 使用 `importlib.metadata.entry_points()` 扫描所有已安装包中名为 `vllm.general_plugins` 的 entry points

2. **导入模块**：对于每个发现的 entry point，vLLM 会导入指定的模块路径

3. **执行注册函数**：vLLM 调用 entry point 指向的函数（如 `register_tq4_backend()`）

4. **注册后端**：注册函数内部调用 `vllm.v1.attention.backends.registry.register_backend()` 将自定义后端注册到 vLLM 的后端注册表中

### 3. 代码流程

```python
# 1. vLLM 扫描 entry points
from importlib.metadata import entry_points
eps = entry_points(group="vllm.general_plugins")

# 2. 遍历并执行每个插件
for ep in eps:
    register_func = ep.load()  # 导入并获取函数
    register_func()  # 执行注册

# 3. 注册函数内部实现（以 TQ4 为例）
def register_tq4_backend():
    from vllm.v1.attention.backends.registry import register_backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_vllm.vllm.tq4_backend.TQ4AttentionBackend",
    )
```

## 手动集成

如果不想使用自动集成，可以手动注册：

```python
from turboquant_vllm.vllm import register_tq4_backend

# 在启动 vLLM 前手动注册
register_tq4_backend()

# 然后启动 vLLM
# vllm serve <model> --attention-backend CUSTOM
```

## 后端选择指南

| 后端 | Entry Point | 计算能力 | 适用场景 |
|------|-------------|----------|----------|
| FlashAttention | `tq4_backend` | 8.0+ | A100, RTX 3090/4090, H100 |
| Triton | `tq4_triton_backend` | 7.5+ | RTX 2080 Ti, T4 等老显卡 |
| FlashInfer | `tq4_flashinfer_backend` | 7.5+ | 需要 FlashInfer 优化场景 |

## 安装与使用

### 安装

```bash
# 基础安装
pip install turboquant-vllm

# 带 vLLM 支持（推荐）
pip install turboquant-vllm[vllm]
```

### 使用

安装后，vLLM 会自动发现并注册所有 TQ4 后端。你只需选择要使用的后端：

```bash
# 使用 FlashAttention 后端（默认，需要 8.0+）
vllm serve <model> --attention-backend CUSTOM

# 使用 Triton 后端（7.5+）
vllm serve <model> --attention-backend TQ4_TRITON

# 使用 FlashInfer 后端（7.5+）
vllm serve <model> --attention-backend TQ4_FLASHINFER
```

## 调试集成

如果自动集成不工作，可以检查：

1. **检查包是否安装**：
   ```bash
   pip show turboquant-vllm
   ```

2. **检查 entry points**：
   ```python
   from importlib.metadata import entry_points
   eps = entry_points(group="vllm.general_plugins")
   for ep in eps:
       print(f"{ep.name}: {ep.value}")
   ```

3. **手动测试注册**：
   ```python
   from turboquant_vllm.vllm import register_tq4_backend
   register_tq4_backend()
   ```

## 添加新的后端

要添加新的后端，需要：

1. **创建后端实现文件**（如 `tq4_new_backend.py`）
2. **实现注册函数**（如 `register_tq4_new_backend()`）
3. **在 `__init__.py` 中导出**
4. **在 `pyproject.toml` 中添加 entry point**：
   ```toml
   [project.entry-points."vllm.general_plugins"]
   tq4_new_backend = "turboquant_vllm.vllm:register_tq4_new_backend"
   ```

## 参考

- [Python Entry Points 文档](https://packaging.python.org/en/latest/specifications/entry-points/)
- [vLLM 插件系统文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)
