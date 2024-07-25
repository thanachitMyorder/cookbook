from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.structured_chat.prompt import SUFFIX
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from tools import edit_image_tool, generate_image_tool

import matplotlib.pyplot as plt
import logging

import chainlit as cl
from chainlit.action import Action
from chainlit.input_widget import Select, Switch, Slider


@cl.action_callback("Create variation")
async def create_variant(action: Action):
    agent_input = f"Create a variation of {action.value}"
    await cl.Message(content=f"Creating a variation of `{action.value}`.").send()
    await main(cl.Message(content=agent_input))


@cl.author_rename
def rename(orig_author):
    mapping = {
        "LLMChain": "Assistant",
    }
    return mapping.get(orig_author, orig_author)


@cl.cache
def get_memory():
    return ConversationBufferMemory(memory_key="chat_history")


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=1,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Steps",
                label="Stability AI - Steps",
                initial=30,
                min=10,
                max=150,
                step=1,
                description="Amount of inference steps performed on image generation.",
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=7,
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            ),
            Slider(
                id="SAI_Width",
                label="Stability AI - Image Width",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
        ]
    ).send()
    await setup_agent(settings)


@cl.on_settings_update
async def setup_agent(settings):
    print("Setup agent with following settings: ", settings)

    llm = ChatOpenAI(
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        model=settings["Model"],
    )
    memory = get_memory()
    _SUFFIX = "Chat history:\n{chat_history}\n\n" + SUFFIX

    agent = initialize_agent(
        llm=llm,
        tools=[generate_image_tool, edit_image_tool],
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={
            "suffix": _SUFFIX,
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cl.user_session.set("generated_image", None)

    # No async implementation in the Stability AI client, fallback to sync
    res = await cl.make_async(agent.run)(
        input=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    elements = []
    actions = []

    generated_image_name = cl.user_session.get("generated_image")
    generated_image = cl.user_session.get(generated_image_name)
    if generated_image:
        elements = [
            cl.Image(
                content=generated_image,
                name=generated_image_name,
                display="inline",
            )
        ]
        actions = [cl.Action(name="Create variation", value=generated_image_name)]

    await cl.Message(content=res, elements=elements, actions=actions).send()

# @cl.on_message
# async def main(message: cl.Message):
#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

#     elements = [
#         cl.Pyplot(name="plot", figure=fig, display="inline"),
#     ]
#     await cl.Message(
#         content="Here is a simple plot",
#         elements=elements,
#     ).send()
@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cl.user_session.set("generated_graph", None)

    # No async implementation in the Stability AI client, fallback to sync
    res = await cl.make_async(agent.run)(
        input=message.content + "assume to format" +  "{\"x_values\": [A1, A2, A3], \"y_values\": [4, 5, 6], \"title\": \"My Graph\", \"xlabel\": \"X\", \"ylabel\": \"Y\"}" , callbacks=[cl.LangchainCallbackHandler()]
    )

    # Log the received message and results
    logging.info(f"Received message: {message.content}")
    logging.info(f"res result from ai: {res}")

    # Create a plot using matplotlib
    fig, ax = plt.subplots()
    ax.plot(["BENZ01", "BENZ02", "BENZ03", "BENZ04"], [10, 20, 30, 40])
    ax.set_title("Simple Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Create elements for the plot
    elements = [
        cl.Pyplot(name="simple_plot", figure=fig, display="inline"),
    ]

    # Send a message with the plot
    await cl.Message(
        content="Here is a simple plot",
        elements=elements,
    ).send()