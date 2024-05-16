import os
import dotenv

dotenv.load_dotenv()

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

def prompt_with_tongyi(template: ChatPromptTemplate, human_input: str) -> str:
    llm = ChatTongyi(
        model_name='qwen-turbo',
        dashscope_api_key=os.getenv('DASHCOPE_API_KEY'),
        temperature=0,
        top_p=0.8,
        streaming=True,
    )

    return llm(template.format_prompt(human_input=human_input).to_messages())


def prompt_with_gemini(template: ChatPromptTemplate, human_input: str) -> str:

    llm = ChatGoogleGenerativeAI(
        model='gemini-pro',
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0,
        top_p=0.8,
        streaming=True,
    )

    return llm(template.format_prompt(human_input=human_input).to_messages())

if __name__ == '__main__':
    role_template = '你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定'

    # CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
    cot_template = """
    作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 
    
    我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
    同时，我也会向客户解释我这样推荐的原因。
    
    示例 1:
      人类：我想找一种象征爱情的花。
      AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。
    
    示例 2:
      人类：我想要一些独特和奇特的花。
      AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
    """
    human_template = "{human_input}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
    system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

    # chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt_cot, human_prompt])

    print(prompt_with_tongyi(chat_prompt, '我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?'))

    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # print(prompt_with_gemini(chat_prompt, '我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?'))


    # Tree of thought demo
    tot_template = """
    设一个顾客在鲜花网站上询问：“我想为我的妻子购买一束鲜花，但我不确定应该选择哪种鲜花。她喜欢淡雅的颜色和花香。”
    
    AI（使用 ToT 框架）：
    
    思维步骤 1：理解顾客的需求。
    顾客想为妻子购买鲜花。
    顾客的妻子喜欢淡雅的颜色和花香。
    
    思维步骤 2：考虑可能的鲜花选择。
    候选 1：百合，因为它有淡雅的颜色和花香。
    候选 2：玫瑰，选择淡粉色或白色，它们通常有花香。
    候选 3：紫罗兰，它有淡雅的颜色和花香。
    候选 4：桔梗，它的颜色淡雅但不一定有花香。
    候选 5：康乃馨，选择淡色系列，它们有淡雅的花香。
    
    思维步骤 3：根据顾客的需求筛选最佳选择。
    百合和紫罗兰都符合顾客的需求，因为它们都有淡雅的颜色和花香。
    淡粉色或白色的玫瑰也是一个不错的选择。
    桔梗可能不是最佳选择，因为它可能没有花香。
    康乃馨是一个可考虑的选择。
    
    思维步骤 4：给出建议。
    “考虑到您妻子喜欢淡雅的颜色和花香，我建议您可以选择百合或紫罗兰。淡粉色或白色的玫瑰也是一个很好的选择。希望这些建议能帮助您做出决策！”
    """
    system_prompt_tot = SystemMessagePromptTemplate.from_template(tot_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt_tot, human_prompt])

    print(prompt_with_tongyi(chat_prompt, '我想学习一种深度学习框架，想入手快并且能快速输出项目成果的.'))


