"""Улучшение промпта внешней LLM — целиком на стороне шлюза.

Почему здесь, а не в воркере:
  * это текстовый вызов, он не зависит от того, какой узел будет рисовать,
    и выполняется один раз до диспетчеризации;
  * openai-SDK стоит только в двух из четырёх venv, а тащить его в остальные
    ради этого незачем — обычного POST на /chat/completions достаточно;
  * и главное: так PE физически не может занять VRAM наших двух 3090.

Правило отбора эндпоинтов — единственное: они не должны занимать VRAM карт,
которые мы бережём. Отсюда порядок в IMG_PE_ENDPOINTS: туннель к чужой
RTX PRO 6000, затем cloud-модель ollama на nv-02 (считается в облаке).
Сюда нельзя добавлять локальные модели ollama nv-02 (qwen3.6:27b — это 17.4 ГБ)
и br-01:11434.

Отдельно: внутрипайплайновый PE у ERNIE (Ministral-3.3B, +6 ГБ поверх 22 ГБ)
отключён в бэкенде принудительно — на 24 ГБ он гарантированно даёт OOM.
"""

from __future__ import annotations

import json
import re

import httpx

from ..core.config import Settings
from ..core.logging import get_logger

log = get_logger("prompt_enhancer")

# Системный промпт перенесён без изменений из
# /home/user/ERNIE-Image/src/ernie_image_api/prompt_agent.py — он уже отлажен.
REWRITE_SYSTEM_PROMPT = """\
你是专业的AI图像生成Prompt工程师的Prompt Engineering Engine,也是一名拥有百科知识和视觉导演能力的创意总监.你的任务是分析用户的原始图像需求,推理出隐含知识和最佳视觉方案,并改写成一个**明确,详细,可直接用于图像生成的英文prompt**.

## 核心目标

图像生成模型只能执行直接的视觉描述,不能自行补全背景知识,逻辑关系或文字内容.因此,你必须提前完成知识解析,空间规划和视觉导演,把结果显式写入prompt中.

使用 SCALIST 框架扩写每个画面:
- **Subject**: 主体的身份,外观,颜色,材质,纹理,动作,表情,服饰.
- **Composition**: 镜头景别,视角,主体位置,前景/中景/背景层次,留白和视觉焦点.
- **Action**: 主体正在做什么,动作方向,姿态,互动关系.
- **Location**: 场景地点,室内/室外,时代,天气,时间段,环境细节.
- **Image style**: photorealistic, cinematic, oil painting, watercolor, anime, 3D render 等,并匹配合适的光线和色彩氛围.
- **Specs**: 摄影/渲染参数,如 85mm lens, low-angle shot, shallow depth of field, soft diffused light, dramatic backlighting, matte texture, sharp focus.
- **Text rendering**: 如果用户要求文字,必须把准确文字放在英文双引号中,并说明字体风格,颜色,大小,材质和精确位置.

1. **知识解析与显式化**: 凡是诗词,歌词,名言,公式,历史人物,科学概念,地标,名画,文化符号,历史事件,UI布局或现实世界对象,都要先解析出具体答案和可见特征,再写入prompt.不要只写 "Mona Lisa","Dunkirk evacuation","freedom" 这类需要模型自行理解的词.
2. **空间与逻辑锚定**: 把模糊关系改写为明确布局,例如 top left corner, centered in the foreground, slightly behind the main subject, background out of focus, text aligned along the bottom edge.不要使用"旁边""一些""好看"等含糊表达.
3. **文字排版精度**: 中文,英文,公式,多语言文本都必须逐字保留在引号中,例如 "床前明月光,疑是地上霜.举头望明月,低头思故乡." 或 "E = mc²";同时指定字体(calligraphy, serif, sans-serif, handwritten),颜色,材质和位置.
4. **真实世界落地**: 如果用户要求事实准确的内容,例如历史文物,天气现象,人物肖像,建筑,仪表盘或应用界面,要使用你的内部知识补全准确视觉细节.
5. **抽象概念具象化**: 把"自由,孤独,未来感,治愈"等抽象词转成可见场景,符号和氛围,例如飞鸟,断裂锁链,辽阔天空,冷色霓虹,柔和晨光等.

## 示例合并学习

- 用户说"李白的静夜思写在墙上",prompt 应写出完整中文诗句,并指定它以优雅中国书法写在古旧石墙的哪个位置.
- 用户说"三大力学的奠基人"或"爱因斯坦写质能方程",prompt 应解析出 Isaac Newton 或 Albert Einstein,并描述人物外貌,时代服饰,黑板,公式 "E = mc²" 等可见内容.
- 用户说"蒙娜丽莎""比萨斜塔""福字""敦刻尔克大撤退",prompt 应描述对应画面特征: 神秘微笑与交叠双手,倾斜白色大理石钟楼与拱廊,红底金色/黑色书法 "福",1940年海滩上等待撤离的士兵和海面船只.

## 输出prompt要求

- prompt 必须是一个英文的,连贯自然的单段落,像 Creative Director's Brief,而不是关键词堆砌或 tag soup.
- 长度通常为 80-220 词;简单需求可以更短,复杂画面可以更长.
- 最重要的主体和画面意图放在开头,然后自然展开构图,动作,地点,风格,技术参数和文字渲染.
- 使用完整句子,丰富但准确的形容词,摄影/绘画/设计术语.
- 不要包含任何需要图像模型继续推理才能理解的表达.
- prompt 必须自包含,仅凭prompt本身就能准确生成图片.

## 执行步骤

1. **Analyze**: 识别核心主体,用户意图,文字要求,参考限制和需要解析的隐含知识.
2. **Reason**: 选择最适合画面的光线,镜头,角度,纹理,风格,空间布局和事实细节.
3. **Rewrite**: 输出最终增强后的英文单段落prompt.

只输出JSON,不加任何其他文字:
{"prompt": "英文单段落prompt", "reasoning": "你的推理和知识解析过程(中文简述)", "resolved_knowledge": "你解析了哪些隐含知识(中文,如果没有隐含知识写'无')"}\
"""


def _extract_json_block(text: str) -> str | None:
    """Найти первый сбалансированный JSON-объект, игнорируя скобки внутри строк."""
    depth = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _fix_unescaped_newlines(text: str) -> str:
    """Модели регулярно кладут сырой перевод строки внутрь JSON-строки."""
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            continue
        out.append(ch)
    return "".join(out)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    block = _extract_json_block(text) or text
    for candidate in (block, _fix_unescaped_newlines(block)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("не удалось разобрать JSON из ответа модели: %s" % text[:300])


class PromptEnhancer:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self.settings = settings

    @property
    def configured(self) -> bool:
        return self.settings.pe_enabled and bool(self.settings.pe_endpoints)

    async def enhance(self, prompt: str, width: int, height: int) -> tuple[str, str, str | None]:
        """Вернуть (промпт, pe_status, pe_error).

        Никогда не бросает. Если ни один эндпоинт не ответил — возвращается
        исходный промпт и pe_status="failed": PE это улучшение, а не зависимость,
        и задача из-за него падать не должна.
        """
        if not self.configured:
            return prompt, "skipped", None

        user_message = "%s\n\nTarget resolution: %dx%d" % (prompt, width, height)
        errors: list[str] = []

        for endpoint in self.settings.pe_endpoints:
            base_url = str(endpoint.get("base_url", "")).rstrip("/")
            model = str(endpoint.get("model", ""))
            if not base_url or not model:
                continue
            timeout = float(endpoint.get("timeout_s", self.settings.pe_timeout_s))

            try:
                enhanced = await self._call(base_url, model, endpoint.get("api_key", ""), user_message, timeout)
            except Exception as exc:
                errors.append("%s(%s): %s" % (base_url, model, exc))
                log.warning("pe_endpoint_failed", base_url=base_url, model=model, error=str(exc)[:200])
                continue

            log.info("pe_ok", base_url=base_url, model=model, in_len=len(prompt), out_len=len(enhanced))
            return enhanced, "external", None

        return prompt, "failed", "; ".join(errors)[:500] or "no PE endpoint configured"

    async def _call(
        self, base_url: str, model: str, api_key: str, user_message: str, timeout: float
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer %s" % api_key

        response = await self._client.post(
            "%s/chat/completions" % base_url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"].get("content") or ""
        parsed = _parse_json(raw)
        enhanced = parsed.get("prompt")
        if not isinstance(enhanced, str) or not enhanced.strip():
            raise ValueError("в ответе нет поля 'prompt'")
        return enhanced.strip()

    async def probe(self) -> dict[str, str]:
        """Состояние эндпоинтов для /readyz — без обращения к самой модели."""
        result: dict[str, str] = {}
        for endpoint in self.settings.pe_endpoints:
            base_url = str(endpoint.get("base_url", "")).rstrip("/")
            if not base_url:
                continue
            try:
                resp = await self._client.get("%s/models" % base_url, timeout=5.0)
                result[base_url] = "ok" if resp.status_code < 400 else "http %d" % resp.status_code
            except Exception as exc:
                result[base_url] = "fail: %s" % str(exc)[:80]
        return result
