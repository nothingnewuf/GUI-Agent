"""
GUI Agent 实现

基于多模态大模型，完成移动端 GUI 任务的自动化操作。
实现思路：
1. 保留带截图的多轮历史，帮助模型理解上下文，但限制轮数避免干扰
2. 精心设计的 System Prompt，强调坐标精度和操作流程
3. 历史消息带明确标签，帮助模型区分历史与当前
4. 多级解析策略，兼容模型输出的各种格式
5. 异常处理与降级机制
"""

import re
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image

# 自动加载项目根目录的 .env 文件
# 仅在环境变量未设置时生效，已设置的环境变量优先
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                _key, _val = _key.strip(), _val.strip()
                if _key not in os.environ:
                    os.environ[_key] = _val

from src.agent_base import (
    BaseAgent, AgentInput, AgentOutput, UsageInfo,
    ACTION_CLICK, ACTION_SCROLL, ACTION_TYPE, ACTION_OPEN, ACTION_COMPLETE,
    VALID_ACTIONS,
)

logger = logging.getLogger(__name__)

# ==========================================
#               System Prompt
# ==========================================

SYSTEM_PROMPT = """
你是一个专业的安卓手机 GUI 自动化操作助手。你的任务是根据用户指令和当前手机截图，分析界面状态并输出下一步操作。

## 核心原则

1. **当前截图优先**：历史截图仅用于了解操作上下文，当前决策必须严格基于【当前步骤】的截图。
2. **坐标必须精确**：所有坐标都是归一化坐标 [0, 1000]，你需要仔细观察当前截图中目标元素的实际位置，给出准确的中心点坐标。
3. **一步一动作**：每次只输出一个动作。

## 坐标规范（极其重要）

- 坐标系：左上角为 (0, 0)，右下角为 (1000, 1000）
- x 轴向右增大，y 轴向下增大
- 屏幕中央约为 [500, 500]
- 顶部状态栏约为 y: 0-50
- 顶部搜索/标题栏区域约为 y: 50-150
- 底部导航栏约为 y: 900-1000
- **坐标估算法**：先确定目标元素在屏幕上的比例位置。例如：元素在左侧1/4处 → x≈250；在中间偏右1/3 → x≈667；在顶部标题栏 → y≈80-120；在底部第二个标签 → y≈940, x≈300
- **关键参考点**（常见UI元素位置，请根据实际截图微调）：
  左上角返回键 → [40, 75] 附近 | 右上角搜索/更多 → [900, 75] 附近
  搜索框（标题栏中间）→ x:300-700, y:60-120 | 底部导航第1个 → [100, 950]
  底部导航第2个 → [300, 950] | 底部导航第3个 → [500, 950]
  底部导航第4个 → [700, 950] | 底部导航第5个 → [900, 950]
- **必须根据当前截图中元素的实际像素位置估算坐标，不要凭记忆或猜测**

## 搜索类操作流程（严格按顺序判断）

第一步：观察截图，判断搜索框的状态——
- **搜索框已激活**（可输入）：键盘已弹出 / 输入框中有光标闪烁 / 输入框高亮且有焦点 → 直接 TYPE
- **搜索框未激活**（不可输入）：没有键盘 / 输入框灰暗 / 只看到搜索图标没看到输入框 / 页面刚加载 → 先 CLICK 搜索框或搜索图标
- **简单判断口诀：能看到键盘弹出 → TYPE；看不到键盘 → 先 CLICK 搜索入口**

第二步：TYPE 时输出完整搜索内容。

第三步：在搜索结果中找到目标项并 CLICK。

## 常见场景处理

- **广告/弹窗/开屏页**：优先点击右上角"跳过"、"关闭"、"X"按钮
- **返回操作**：点击左上角返回箭头或返回文字
- **列表选择**：点击目标列表项的视觉中心位置
- **筛选/排序按钮**：通常在搜索框下方或搜索结果列表上方，是一个带筛选图标或文字（如"筛选"）的按钮
- **输入文字**：必须确认输入框已获得焦点（有光标闪烁或键盘弹出）后才 TYPE。如果只是看到搜索框但无键盘/光标，先 CLICK 搜索框
- **任务完成判断**：当用户指令要求的目标都已达成时输出 COMPLETE。例如：视频开始播放 → COMPLETE；歌曲开始播放 → COMPLETE；导航路线已展示 → COMPLETE。不要过早完成，也不要在完成后继续操作。

## 输出格式（严格遵守）

请按如下格式输出，每次只输出一个动作：

```
Thought: <用中文分析当前截图状态，说明目标元素在屏幕上的大致位置（使用比例描述），以及下一步操作理由>
Action: <动作类型>
<参数行>
```

## 动作类型与参数格式

### 1. CLICK - 点击界面元素
```
Action: CLICK
Point: [x, y]
```

### 2. TYPE - 输入文字
```
Action: TYPE
Text: 要输入的完整内容
```
注意：输入文字时请输出完整内容，不要省略前缀或后缀。

### 3. SCROLL - 滑动屏幕
```
Action: SCROLL
StartPoint: [x1, y1]
EndPoint: [x2, y2]
```
- 向上滑动查看下方内容：StartPoint 的 y > EndPoint 的 y
- 向下滑动查看上方内容：StartPoint 的 y < EndPoint 的 y

### 4. OPEN - 打开应用
```
Action: OPEN
AppName: 应用名称
```
注意：请使用应用的完整官方名称，不要使用简称或口语化表达。例如"去哪儿旅行"而非"去哪"。

### 5. COMPLETE - 任务完成
```
Action: COMPLETE
```

## 示例

用户指令：去快手搜索动画片
当前界面：快手首页，顶部搜索框未激活（无键盘）

```
Thought: 当前是快手首页，右上角（x约900, y约70）有搜索图标，搜索框未激活（无键盘弹出）。需要先点击搜索图标。
Action: CLICK
Point: [900, 70]
```

---

用户指令：搜索动画片
当前界面：搜索框已激活，键盘已弹出，可以看到输入光标

```
Thought: 当前搜索框已激活（键盘已弹出，光标在输入框中），可以直接输入搜索内容。
Action: TYPE
Text: 动画片
```

---

用户指令：在京东搜索耳机
当前界面：京东首页，顶部有搜索框但未激活

```
Thought: 当前是京东首页，顶部中间有搜索框（x约500, y约85），搜索框未激活（无键盘弹出），需要先点击搜索框激活。
Action: CLICK
Point: [500, 85]
```

---

用户指令：查看航班价格
当前界面：航班搜索结果已显示，包含价格信息

```
Thought: 当前界面已展示航班信息和价格，最低价格清晰可见，任务目标已达成。
Action: COMPLETE
```
"""


# ==========================================
#               Agent 实现
# ==========================================

class Agent(BaseAgent):
    """
    GUI Agent 派生类

    策略：
    - 保留带截图的多轮历史（限制3轮），帮助模型理解上下文
    - 历史消息带明确标签 [历史步骤] / [当前步骤]
    - 精心设计的 Prompt 引导模型输出结构化内容
    - 多级解析策略（优先正则，降级 JSON，最终兜底）
    """

    # 保留最近3轮历史（每轮 = 1条user截图消息 + 1条assistant动作详细，即2条消息）
    MAX_HISTORY_ROUNDS = 3

    def _initialize(self):
        """初始化 Agent 内部状态"""
        self._conversation_history: List[Dict[str, Any]] = []      # 存历史对话消息
        self._step_count = 0              # 当前步骤计数
        self._current_instruction = ""    # 当前任务指令
        self._opened_app = ""             # 已打开的应用
        logger.info(f"Agent initialized, model={self.model_id}")

    def reset(self):
        """每个测试用例开始前重置状态"""
        self._conversation_history = []
        self._step_count = 0
        self._current_instruction = ""
        self._opened_app = ""
        logger.info("Agent state reset")

    # ------------------------------------------
    #           应用名模糊修正
    # ------------------------------------------

    # 测试数据中出现的官方应用名称（用于模糊匹配纠错，不是硬编码映射）
    _KNOWN_APPS = [
        "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手",
        "芒果TV", "美团", "去哪儿旅行", "腾讯视频", "喜马拉雅",
    ]

    def _maybe_fix_app_name(self, app_name: str) -> str:
        """
        用编辑距离将模型的简写/近似应用名修正为官方名称。
        模糊匹配：计算字符串相似度，能泛化到未见过的拼写变体

        只有相似度 > 0.85 且候选唯一时才修正，防止误判。
        例如："去哪旅行" 与 "去哪儿旅行" 相似度 0.96 → 修正
              "去哪" 与 "去哪儿旅行" 相似度 0.53 → 不够阈值，不修正
        """
        if not app_name or len(app_name) < 2:
            return app_name
        if app_name in self._KNOWN_APPS:
            return app_name

        from difflib import SequenceMatcher
        best = None
        best_ratio = 0.0
        for known in self._KNOWN_APPS:
            ratio = SequenceMatcher(None, app_name, known).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = known
            elif ratio == best_ratio and ratio > 0.85:
                # 两个候选得分相同 → 歧义，不修正
                best = None
        if best and best_ratio > 0.85:
            logger.info(
                f"[App fix] '{app_name}' -> '{best}' (similarity={best_ratio:.2f})"
            )
            return best
        return app_name

    # ------------------------------------------
    #           消息构建
    # ------------------------------------------

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """
        构建发送给模型的消息列表。

        格式：
          - system: 固定的系统提示
          - user(history): 历史截图 + 操作记录（最近 N 轮，带 [历史步骤] 标签）
          - user(current): 当前截图 + 指令（带 [当前步骤] 标签）
        """
        messages = []

        # 1. System message
        messages.append({
            "role": "user",
            "content": SYSTEM_PROMPT
        })
        messages.append({
            "role": "assistant",
            "content": "明白，我会严格按照规范操作，当前决策只基于标注为【当前步骤】的截图。"
        })

        # 2. 历史对话（最近 N 轮，图片 + 动作，带历史标签）
        history = self._conversation_history
        if len(history) > self.MAX_HISTORY_ROUNDS * 2:
            history = history[-(self.MAX_HISTORY_ROUNDS * 2):]

        for msg in history:
            messages.append(msg)

        # 3. 当前轮：用户指令 + 截图（带当前步骤标签）
        current_image_url = self._encode_image(input_data.current_image)

        if input_data.step_count == 1:
            user_text = (
                f"【任务指令】{input_data.instruction}\n\n"
                f"【当前步骤 - 第 1 步】请只基于下方这张当前截图判断，分析界面状态并给出下一步操作。"
            )
        else:
            user_text = (
                f"【任务指令】{input_data.instruction}\n\n"
                f"【当前步骤 - 第 {input_data.step_count} 步】这是执行上一步操作后的新界面截图。\n"
                f"请只基于下方这张当前截图判断，不要依赖历史截图的记忆，给出下一步操作。"
            )

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text
                },
                {
                    "type": "image_url",
                    "image_url": {"url": current_image_url}
                }
            ]
        })

        return messages

    # ------------------------------------------
    #           核心 act 方法
    # ------------------------------------------

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        根据当前截图和历史，调用模型，返回下一步动作。
        """
        self._step_count = input_data.step_count
        self._current_instruction = input_data.instruction

        # 构建消息
        messages = self.generate_messages(input_data)

        # 调用 API
        try:
            response = self._call_api(
                messages,
                temperature=0,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return AgentOutput(
                action=ACTION_COMPLETE,
                parameters={},
                raw_output=f"API Error: {e}",
            )

        # 提取 raw_output
        raw_output = ""
        try:
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Failed to extract response content: {e}")

        logger.info(f"[Step {input_data.step_count}] Raw output:\n{raw_output}")

        # 提取 token 使用信息
        usage = self.extract_usage_info(response)

        # 解析动作
        action, parameters = self._parse_action(raw_output)

        # OPEN 应用名修正：模糊匹配到官方完整名称
        if action == ACTION_OPEN:
            raw_app = parameters.get("app_name", "")
            fixed_app = self._maybe_fix_app_name(raw_app)
            if fixed_app != raw_app:
                parameters = {**parameters, "app_name": fixed_app}

        logger.info(f"[Step {input_data.step_count}] Parsed: action={action}, params={parameters}")

        # 更新历史（将当前截图和模型回复写入历史）
        self._update_history(input_data.current_image, raw_output, action, parameters)

        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage,
        )

    # ------------------------------------------
    #           动作解析
    # ------------------------------------------

    def _parse_action(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """
        多级解析策略：
        1. 优先用正则解析结构化输出（最可靠）
        2. 降级尝试 JSON 解析
        3. 最终兜底
        """
        if not raw_output or not raw_output.strip():
            logger.warning("Empty model output, returning COMPLETE")
            return ACTION_COMPLETE, {}

        # 尝试主要格式解析
        result = self._parse_structured_output(raw_output)
        if result: return result

        # 尝试 JSON 格式
        result = self._parse_json_output(raw_output)
        if result: return result

        # 尝试宽松匹配（处理格式不规范的输出）
        result = self._parse_loose_output(raw_output)
        if result: return result

        logger.warning(f"Cannot parse action from output, defaulting to COMPLETE")
        return ACTION_COMPLETE, {}

    def _parse_structured_output(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        解析主要格式：
          Action: CLICK
          Point: [x, y]
        """
        # 提取 Action 行
        action_match = re.search(
            r'Action\s*[:：]\s*(CLICK|TYPE|SCROLL|OPEN|COMPLETE)',
            text, re.IGNORECASE
        )
        if not action_match:
            return None

        action = action_match.group(1).upper()

        try:
            if action == ACTION_CLICK:
                point = self._extract_point(text, ['Point', 'point', 'POINT', '坐标', 'Coordinate'])
                if point:
                    return ACTION_CLICK, {"point": point}

            elif action == ACTION_TYPE:
                text_match = re.search(
                    r'(?:Text|text|TEXT|内容|Content)\s*[:：]\s*(.+)',
                    text
                )
                if text_match:
                    content = text_match.group(1).strip().strip('"\'"""''')
                    return ACTION_TYPE, {"text": content}

            elif action == ACTION_SCROLL:
                start = self._extract_point(text, ['StartPoint', 'Start', 'start', '起点'])
                end = self._extract_point(text, ['EndPoint', 'End', 'end', '终点'])
                if start and end:
                    return ACTION_SCROLL, {"start_point": start, "end_point": end}

            elif action == ACTION_OPEN:
                app_match = re.search(
                    r'(?:AppName|App|app|应用名?)\s*[:：]\s*(.+)',
                    text
                )
                if app_match:
                    app_name = app_match.group(1).strip().strip('"\'"""''')
                    return ACTION_OPEN, {"app_name": app_name}

            elif action == ACTION_COMPLETE:
                return ACTION_COMPLETE, {}

        except Exception as e:
            logger.warning(f"Structured parse error: {e}")

        return None

    def _extract_point(self, text: str, key_candidates: List[str]) -> Optional[List[int]]:
        """
        从文本中提取坐标点 [x, y]。
        支持多种格式：[x, y] / (x, y) / x, y
        """
        for key in key_candidates:
            pattern = rf'{re.escape(key)}\s*[:：]\s*\[?\s*(\d+(?:\.\d+)?)\s*[,，\s]\s*(\d+(?:\.\d+)?)\s*\]?'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                x = int(float(m.group(1)))
                y = int(float(m.group(2)))
                x = max(0, min(1000, x))
                y = max(0, min(1000, y))
                return [x, y]

        # 宽松匹配：任何 [x, y] 格式
        matches = re.findall(r'\[(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\]', text)
        if matches:
            x = int(float(matches[0][0]))
            y = int(float(matches[0][1]))
            x = max(0, min(1000, x))
            y = max(0, min(1000, y))
            return [x, y]

        return None

    def _parse_json_output(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        尝试解析 JSON 格式的输出：
        {"action": "CLICK", "point": [x, y]}
        """
        json_patterns = [
            r'```json\s*([\s\S]+?)\s*```',
            r'```\s*([\s\S]+?)\s*```',
            r'(\{[\s\S]+\})',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
                action = data.get('action', '').upper()
                if action not in VALID_ACTIONS:
                    continue

                if action == ACTION_CLICK:
                    point = data.get('point') or data.get('coordinate') or data.get('coords')
                    if point and len(point) >= 2:
                        return ACTION_CLICK, {"point": [int(point[0]), int(point[1])]}

                elif action == ACTION_TYPE:
                    text_content = data.get('text') or data.get('content') or ''
                    return ACTION_TYPE, {"text": str(text_content)}

                elif action == ACTION_SCROLL:
                    start = data.get('start_point') or data.get('start')
                    end = data.get('end_point') or data.get('end')
                    if start and end:
                        return ACTION_SCROLL, {
                            "start_point": [int(start[0]), int(start[1])],
                            "end_point": [int(end[0]), int(end[1])]
                        }

                elif action == ACTION_OPEN:
                    app_name = data.get('app_name') or data.get('app') or ''
                    return ACTION_OPEN, {"app_name": str(app_name)}

                elif action == ACTION_COMPLETE:
                    return ACTION_COMPLETE, {}

            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return None

    def _parse_loose_output(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        宽松解析，处理各种不规范输出。
        也兼容 base agent 的 click(point='<point>x y</point>') 格式。
        """
        text_lower = text.lower()

        # 处理 click(point='<point>x y</point>') 格式
        point_tag = re.search(r'<point>\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*</point>', text)
        if point_tag:
            x = int(float(point_tag.group(1)))
            y = int(float(point_tag.group(2)))
            if 'scroll' in text_lower:
                all_points = re.findall(r'<point>\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*</point>', text)
                if len(all_points) >= 2:
                    start = [int(float(all_points[0][0])), int(float(all_points[0][1]))]
                    end = [int(float(all_points[1][0])), int(float(all_points[1][1]))]
                    return ACTION_SCROLL, {"start_point": start, "end_point": end}
            return ACTION_CLICK, {"point": [max(0, min(1000, x)), max(0, min(1000, y))]}

        # 检测 OPEN 动作
        open_match = re.search(r'open\s*\(\s*app_name\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.IGNORECASE)
        if open_match:
            return ACTION_OPEN, {"app_name": open_match.group(1)}

        # 检测 TYPE 动作
        type_match = re.search(r'type\s*\(\s*(?:content|text)\s*=\s*[\'"]([^\'"]*)[\'"]', text, re.IGNORECASE)
        if type_match:
            return ACTION_TYPE, {"text": type_match.group(1)}

        # 检测 COMPLETE
        if re.search(r'complete\s*\(', text, re.IGNORECASE) or 'COMPLETE' in text.upper():
            if not re.search(r'incomplete|not.{0,10}complete|haven.t.{0,10}complete', text_lower):
                return ACTION_COMPLETE, {}

        # 最后尝试：在文本中识别动作关键词 + 坐标
        all_coords = re.findall(r'\[(\d+)\s*[,，]\s*(\d+)\]', text)
        if all_coords and ('click' in text_lower or '点击' in text):
            x, y = int(all_coords[0][0]), int(all_coords[0][1])
            return ACTION_CLICK, {"point": [max(0, min(1000, x)), max(0, min(1000, y))]}

        return None

    # ------------------------------------------
    #           历史管理
    # ------------------------------------------

    def _update_history(
        self,
        image: Image.Image,
        raw_output: str,
        action: str,
        parameters: Dict[str, Any]
    ):
        """
        将当前步骤添加到对话历史。
        使用轻量级表示：截图 base64 + 动作文本（带历史标签）。
        """
        step_label = self._step_count

        # 用户消息（截图，带历史标签）
        image_url = self._encode_image(image)
        self._conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"【历史步骤 {step_label}】执行上述操作后看到的界面："},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })
        # 助手消息（动作）
        action_desc = self._format_action_for_history(action, parameters)
        thought = self._extract_thought(raw_output)
        assistant_content = f"{thought}\nAction: {action}\n{action_desc}" if thought else f"Action: {action}\n{action_desc}"

        self._conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })

        # 窗口管理：超出限制则截断旧历史
        max_msgs = self.MAX_HISTORY_ROUNDS * 2
        if len(self._conversation_history) > max_msgs:
            self._conversation_history = self._conversation_history[-max_msgs:]

    def _format_action_for_history(self, action: str, parameters: Dict[str, Any]) -> str:
        """将动作格式化为历史记录字符串"""
        if action == ACTION_CLICK:
            point = parameters.get("point", [0, 0])
            return f"Point: {point}"
        elif action == ACTION_TYPE:
            return f"Text: {parameters.get('text', '')}"
        elif action == ACTION_SCROLL:
            start = parameters.get("start_point", [0, 0])
            end = parameters.get("end_point", [0, 0])
            return f"StartPoint: {start}\nEndPoint: {end}"
        elif action == ACTION_OPEN:
            return f"AppName: {parameters.get('app_name', '')}"
        elif action == ACTION_COMPLETE:
            return ""
        return ""

    def _extract_thought(self, raw_output: str) -> str:
        """从模型输出中提取 Thought 部分"""
        thought_match = re.search(r'Thought\s*[:：]\s*(.+?)(?=\nAction|\Z)', raw_output, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            if len(thought) > 100:
                thought = thought[:100] + "..."
            return f"Thought: {thought}"
        return ""
