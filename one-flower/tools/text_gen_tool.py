from langchain.chains.llm import LLMChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate

# 导入所需的类
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field
from typing import List


# 定义一个名为TextParsing的模型，描述了如何解析大V信息
class TextParsing(BaseModel):
    summary: str = Field(description="大V个人简介")  # 大V的简介或背景信息
    facts: List[str] = Field(description="大V的特点")  # 大V的一些显著特点或者事实
    interests: List[str] = Field(description="这个大V可能感兴趣的事情")  # 大V可能感兴趣的主题或活动
    letter: List[str] = Field(description="一篇联络这个大V的邮件")  # 联络大V的建议邮件内容

    # 将模型对象转换为字典
    def to_dict(self):
        return {
            "summary": self.summary,
            "facts": self.facts,
            "interests": self.interests,
            "letter": self.letter,
        }


# 创建一个基于Pydantic模型的解析器，用于将文本输出解析为特定的结构
letter_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=TextParsing
)


def generate_letter(llm: BaseLLM, information: str):
    letter_template = """
         下面是这个人的微博信息 {information}
         请你帮我:
         1. 写一个简单的总结
         2. 挑两件有趣的特点说一说
         3. 找一些他比较感兴趣的事情
         4. 写一篇热情洋溢的介绍信
         {format_instructions}
     """
    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=letter_template,
        partial_variables={"format_instructions": letter_parser.get_format_instructions()}
    )

    # 初始化链
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # 生成文案
    response = chain.invoke({"information": information})
    to_dict = letter_parser.parse(response.get("text")).to_dict()

    return to_dict
    # retry_parser = RetryWithErrorOutputParser.from_llm(parser=letter_parser, llm=llm)
    # parsed_result = retry_parser.parse_with_prompt(response, prompt_template.format_prompt(information=information))
    #
    # return parsed_result.to_dict()
