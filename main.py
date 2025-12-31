from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv(override=True)

DEEPSEEK_API = os.getenv("DEEPSEEK_API")

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=DEEPSEEK_API,
    # other params...
)

from langchain.messages import SystemMessage, HumanMessage, AIMessage
system_message = "你叫小智，是一名乐于助人的智能助手。请在对话中保持温和，有耐心的语气。"
system_message = SystemMessage(content=system_message)

messages = [
    system_message
]


agent = create_agent(
    model=model,
    system_prompt="你是一名多才多艺的智能助手，可以调用工具帮助用户解决问题。",
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 20),
            keep=("messages", 2),
            summary_prompt="请简要总结以下对话内容，保留关键信息，以便后续使用：\n{messages}",
        )
    ],
)

messages.append(HumanMessage(content="我叫陈明"))
result = agent.invoke({
    "messages": messages,
})
print(result)
messages.append(AIMessage(content=result["messages"][-1].content))
messages.append(HumanMessage(content="我20岁"))
result = agent.invoke({
    "messages": messages,
})
print(result)
messages.append(AIMessage(content=result["messages"][-1].content))
messages.append(HumanMessage(content="我希望你叫小天"))
result = agent.invoke({
    "messages": messages,
})
print(result)
messages.append(AIMessage(content=result["messages"][-1].content))
messages.append(HumanMessage(content="你叫什么"))
result = agent.invoke({
    "messages": messages,
})
print(result)
