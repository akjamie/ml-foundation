import asyncio
import os

import dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.globals import set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler, CallbackManager
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

set_debug(True)

dotenv.load_dotenv()
from typing import Any, Dict, List

from langchain_community.chat_models.tongyi import ChatTongyi


# 创建同步回调处理器
class MyFlowerShopSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"获取花卉数据: token: {token}")


# 创建异步回调处理器
class MyFlowerShopAsyncHandler(AsyncCallbackHandler):

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("正在获取花卉数据...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("花卉数据获取完毕。提供建议...")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("整理花卉建议...")
        await asyncio.sleep(0.5)  # 模拟异步操作
        print("祝你今天愉快！")


# 主要的异步函数
async def main():
    flower_shop_chat = ChatTongyi(
        dashscope_api_key=os.environ['DASHSCOPE_API_KEY'],
        max_tokens=100,
        streaming=True,
        callbacks=[MyFlowerShopSyncHandler(), MyFlowerShopAsyncHandler()],
    )

    # 异步生成聊天回复
    await flower_shop_chat.agenerate([[HumanMessage(content="哪种花卉最适合生日？只简单说3种，不超过50字")]])


# 运行主异步函数
# asyncio.run(main())


class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        # Calculation I
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

        # Calculation II
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokens = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self.input_tokens += sum(len(prompt.split()) for prompt in prompts)

    def on_llm_new_token(self, token: str, **kwargs):
        self.output_tokens += len(token)

    def on_llm_end(self, response: dict, **kwargs):
        # Extracting token usage directly from the first (and in this case, only) generation
        generations = response.generations
        if None != generations:
            first_generation = generations[0][0]
            gen_info = first_generation.generation_info

            # Updating total token counts
            self.total_prompt_tokens += gen_info.get('token_usage')["input_tokens"]
            self.total_completion_tokens += gen_info.get('token_usage')["output_tokens"]

            # Updating dictionary of total tokens
            self.total_tokens["prompt_tokens"] = self.total_prompt_tokens
            self.total_tokens["completion_tokens"] = self.total_completion_tokens
            self.total_tokens["total_tokens"] = self.total_completion_tokens + self.total_prompt_tokens

        # Alternatively, if token_usage is directly under llm_output, handle that too
        token_usage_from_llm_output = (response.llm_output or {}).get("token_usage", {})
        self.total_prompt_tokens += token_usage_from_llm_output.get("input_tokens", 0)
        self.total_completion_tokens += token_usage_from_llm_output.get("output_tokens", 0)
        self.total_tokens["prompt_tokens"] = self.total_prompt_tokens
        self.total_tokens["completion_tokens"] = self.total_completion_tokens
        self.total_tokens["total_tokens"] = self.total_completion_tokens + self.total_prompt_tokens

        self.tokens = self.input_tokens + self.output_tokens

        print("=========Calc I=========")
        print(f"Total input tokens: {self.total_prompt_tokens}")
        print(f"Total output tokens: {self.total_completion_tokens}")
        print(f"Total tokens used: {self.total_tokens}")

        print("=========Calc II=========")
        print(f"Total input tokens: {self.input_tokens}")
        print(f"Total output tokens: {self.output_tokens}")
        print(f"Total tokens used: {self.tokens}")


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler

# 使用context manager进行token counting
handler = TokenCounter()
callback_manager = CallbackManager([handler])

# 初始化大语言模型
llm = ChatTongyi(temperature=0.5, dashscope_api_key=os.environ['DASHSCOPE_API_KEY'], callbacks=callback_manager,
                 streaming=True)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

with get_openai_callback() as cb:
    # 第一天的对话
    # 回合1
    conversation("我姐姐明天要过生日，我需要一束生日花束。")
    print("第一次对话后的记忆:", conversation.memory.buffer)

    # 回合2
    conversation("她喜欢粉色玫瑰，颜色是粉色的。")
    print("第二次对话后的记忆:", conversation.memory.buffer)

    # 回合3 （第二天的对话）
    conversation("我又来了，还记得我昨天为什么要来买花吗？")
    print("/n第三次对话后时提示:/n", conversation.prompt.template)
    print("/n第三次对话后的记忆:/n", conversation.memory.buffer)

# 输出使用的tokens
print("\n总计使用的tokens:", cb.total_tokens)
