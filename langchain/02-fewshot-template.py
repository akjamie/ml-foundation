import os
import dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

dotenv.load_dotenv()
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

def prompt_with_tongyi(prompt: str, flower_type: str, occasion: str) -> str:
    llm = Tongyi(
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        model_name="qwen-turbo",
        temperature=0.5,
        top_p=0.8,
        streaming=True,
    )
    return llm.invoke(prompt.format(flower_type=flower_type, occasion=occasion))


if __name__ == "__main__":
    # 1. 创建一些示例
    samples = [
        {
            "flower_type": "玫瑰",
            "occasion": "爱情",
            "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
        },
        {
            "flower_type": "康乃馨",
            "occasion": "母亲节",
            "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
        },
        {
            "flower_type": "百合",
            "occasion": "庆祝",
            "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
        },
        {
            "flower_type": "向日葵",
            "occasion": "鼓励",
            "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
        }
    ]
    template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
    prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"],
                                   template=template)
    prompt = FewShotPromptTemplate(
        examples=samples,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
    )
    # print(prompt.format(flower_type="野玫瑰", occasion="爱情"))


    # result = prompt_with_tongyi(prompt, "野玫瑰", "爱情")
    # print(result)


    # 5. 使用示例选择器

    # 初始化示例选择器
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        samples,
        DashScopeEmbeddings(dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]),
        Chroma,
        k=1
    )

    # 创建一个使用示例选择器的FewShotPromptTemplate对象
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}",
        input_variables=["flower_type", "occasion"]
    )
    print(prompt.format(flower_type="红玫瑰", occasion="爱情"))
    result = prompt_with_tongyi(prompt, "野玫瑰", "爱情")
    print(result)
