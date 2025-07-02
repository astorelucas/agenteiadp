from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import HumanMessage
from agentai.tools import wiki_tool, save_tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

load_dotenv()

# -----Chat model Instantiation ----
# llm = ChatAnthropic(model= "claude-3-7-sonnet-20250219")
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# chat_model = ChatHuggingFace(llm=llm)

chat = ChatDeepInfra(model="meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
response=chat.invoke(messages)
print(response)
# ---- Invocation -----
# # response = llm.invoke("what is the meaning of life?")
# # print(response)
# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(
#         content="I'm Lucas Malacarne Astore, do you know me ?"
#     ),
# ]
# ai_msg = chat_model.invoke(messages)
# print(ai_msg.content)


# class ResearchResponse(BaseModel):
#     topic: str
#     summary: str
#     source: list[str]
#     tools_used: list[str]

# parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# prompt = ChatPromptTemplate.from_messages(

#     [
#         (
#             "system",
#             """
#             You are a research assistant that will help generate a research paper.
#             Answer the user query and use neccessary tools.
#             Wrap the output i this format and provide no other text \n{format_instructions}
#             """,
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{query}"),
#         ("placeholder", "{agent_scratchpad}")
#     ]
# ).partial(format_instructions=parser.get_format_instructions())

# tools = [wiki_tool, save_tool]

# # chat_with_tools = chat_model.bind_tools(tools)
# # ai_msg = chat_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
# # ai_msg.tool_calls

# agent = create_tool_calling_agent(llm=chat_model, tools=tools, prompt=prompt)

# # agent = create_tool_calling_agent(
# #     llm=llm,
# #     prompt= prompt,
# #     tools=tools
# # )

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # query = input('What can I help you research?')
# query = "what is the capital of united states?"

# raw_response = agent_executor.invoke({"query":query})

# for key, value in raw_response.items():
#     print(f"Key: {key}, Value: {value}")

# try:
#     structured_response = parser.parse(raw_response.get("output")[0]["text"])
#     print(structured_response)
# except Exception as e:
#     print("Error parsing reponse", e,"Raw response", raw_response)
