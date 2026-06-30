# 工程方法论参考 — AI 智能体开发的物理隔离手段

## 状态机设计模式

### 核心思路
将智能体的工作流硬编码成一张有向图：
- **节点** = 具体动作（提取意图、查数据库、生成回复）
- **边** = 条件转移（工具返回空→走节点A；返回数据→走节点B）

### Python 示例（精简版）
```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class State(Enum):
    ANALYZE_INPUT = "analyze_input"
    CALL_TOOL = "call_tool"
    REFLECT = "reflect"
    GENERATE = "generate"
    ERROR = "error"
    DONE = "done"

@dataclass
class Transition:
    next_state: State
    action: str

class AgentFSM:
    def __init__(self):
        self.state = State.ANALYZE_INPUT
        self.data: dict[str, Any] = {}
    
    async def step(self):
        if self.state == State.ANALYZE_INPUT:
            # 提取意图
            result = await analyze_user_input(self.data["user_msg"])
            self.data["intent"] = result
            next_state = State.CALL_TOOL if result.get("needs_tool") else State.GENERATE
            return Transition(next_state, "analyzed")
        
        elif self.state == State.CALL_TOOL:
            result = await execute_tool(self.data["intent"]["tool"], self.data["intent"]["args"])
            self.data["tool_result"] = result
            if result.get("success"):
                return Transition(State.GENERATE, "tool_ok")
            else:
                return Transition(State.ERROR, "tool_failed")
        
        elif self.state == State.GENERATE:
            reply = await generate_reply(self.data)
            self.data["reply"] = reply
            return Transition(State.DONE, "generated")
        
        elif self.state == State.ERROR:
            reply = f"遇到问题：{self.data.get('tool_result', {}).get('error', '未知错误')}"
            self.data["reply"] = reply
            return Transition(State.DONE, "error_handled")
```

## 契约驱动开发

### Pydantic Schema 示例
```python
from pydantic import BaseModel, Field
from typing import Optional

class ToolInput(BaseModel):
    """所有工具输入的基类"""
    session: str = Field(default="default", description="会话ID")

class WebSearchInput(ToolInput):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)

class WebSearchOutput(BaseModel):
    success: bool
    results: list[dict] = Field(default_factory=list)
    error: Optional[str] = None
    total_results: int = Field(default=0, ge=0)

# 使用例
input_data = WebSearchInput(query="今天AI新闻", session="test")
output_data = WebSearchOutput(success=True, results=[], total_results=0)
```

### 主循环中的契约校验
```python
async def execute_tool_safe(name: str, args: dict) -> dict:
    """安全执行工具，带 schema 校验"""
    try:
        # 查找对应工具的 schema
        schema = TOOL_SCHEMAS.get(name)
        if schema:
            validated = schema(**args)  # 自动校验
            args = validated.model_dump()
        result = await execute_tool(name, args)
        return {"success": True, "result": result}
    except ValidationError as e:
        return {"success": False, "error": f"参数校验失败: {e}"}
```

## 微内核 + 插件化架构

### 内核职责（只做三件事）
```python
class Kernel:
    """智能体内核：极简稳定"""
    
    def __init__(self):
        self.state = {}      # 1. 维护当前状态
        self.plugins = {}    # 2. 注册的插件
        self.scheduler = Scheduler()  # 3. 任务调度
    
    def register_plugin(self, name: str, plugin):
        """注册插件，不修改内核代码"""
        self.plugins[name] = plugin
    
    async def route(self, intent: str, args: dict):
        """路由任务到对应插件"""
        plugin = self.plugins.get(intent)
        if not plugin:
            return {"error": f"未知意图: {intent}"}
        return await plugin.execute(args)
```

### 插件接口
```python
class Plugin(ABC):
    """所有插件必须实现的接口"""
    
    @abstractmethod
    async def execute(self, args: dict) -> dict:
        pass
    
    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        """返回输入校验 schema"""
        pass
```

## Eval 黄金数据集

### 结构
```python
EVAL_DATASET = [
    {
        "input": "帮我查一下今天的天气",
        "expected_behavior": ["should_call_tool:get_weather"],
        "expected_response_contains": ["天气"],
        "variations": ["今天天气怎么样", "查天气", "weather today"]
    },
    {
        "input": "写一个Python冒泡排序",
        "expected_behavior": ["should_generate_code"],
        "expected_response_contains": ["def", "bubble_sort"],
        "check_not_contains": ["以下是代码"]  # 避免过度模板化回复
    },
    # ... 持续积累历史失败 case
]
```

### 自动跑 Eval
```python
async def run_eval(model, dataset: list) -> dict:
    """对每个 case 打分，汇总准确率"""
    scores = []
    for case in dataset:
        response = await model(case["input"])
        score = evaluate_response(response, case)
        scores.append(score)
    
    accuracy = sum(scores) / len(scores)
    return {
        "accuracy": accuracy,
        "total": len(scores),
        "passed": sum(scores),
        "failed": len(scores) - sum(scores),
    }
```
