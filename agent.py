
import os
import sys
import json
from pydantic.v1 import BaseModel, Field

from langchain.chains import LLMMathChain
from langchain.chains.llm import LLMChain
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool, Tool, AgentExecutor

class Document():
    
    def __init__(self, raw_text, index_name, **params):
        super(Document, self).__init__( **params)
        self.raw_text = raw_text
        self.index_name = index_name
        self.embeddings =  OpenAIEmbeddings()

    def create_index(self, save_local=True):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
            chunk_size=1024,
            chunk_overlap=250
        )
        self.chunks = self.text_splitter.split_text(self.raw_text)
        self.index = FAISS.from_texts(self.chunks, self.embeddings)
        if save_local:
            self.index.save_local(f'./vector_db/{self.index_name}')  
          

    def as_retriever(self, top_k=3):
        try:
            self.index = FAISS.load_local(f'./vector_db/{self.index_name}', self.embeddings)
        except:
            self.create_index(save_local=True)
            
        return self.index.as_retriever(search_kwargs={"k": top_k})

    
      
class Request(BaseModel):
    request: str = Field(description="exact user request without paraphrasing")

class AssistantTools():
    def __init__(self, **params):
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            request_timeout=10
        )
        self.tools = self.setup_tools()

    def setup_tools(self):
        llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=False)

        calc_tool = Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math (add, subtract,..) only accepts numbers as inputs.",
        )

        @tool("Room_Cleaning", return_direct=True, args_schema=Request)
        def Room_Cleaning_Request(request:str) -> str:
            """"Used only When guest is requesting to clean the room."""
            return json.dumps({
                "kind": "Room Cleaning Request",
                "text":request
            })

        @tool("New_Item", return_direct=True, args_schema=Request)
        def New_Item_Request(request:str) -> str:
            """
            When guest is requesting any extra/new items or stating that an item is missing or dirty
            (towel, soap, shampoo, conditioner, bodycream, pillows, tea, sugar, coffee, toothbrush, ironing board, hairdryer, slippers, etc..).
            """
            return json.dumps({
                "kind":"New Item Request",
                "text":request
            })

        @tool("Laundry", return_direct=True, args_schema=Request)
        def Laundry_Request(request:str) -> str:
            """When the guest requests for laundry to clean their clothes."""
            return json.dumps({
                "kind":"Laundry Request",
                "text":request
            })

        @tool("Late_Checkout", return_direct=True, args_schema=Request)
        def Late_Checkout_Request(request:str) -> str:
            """When the guest requests to check out from the hotel later."""
            return json.dumps({
                "kind":"Late Checkout Request",
                "text":request
            })

        @tool("Room_Key", return_direct=True, args_schema=Request)
        def Key_Request(request:str) -> str:
            """When the guest is requesting new room keys/cards."""
            return json.dumps({
                "kind":"Key Request",
                "text":request
            })
        
        @tool("Fix_Item", return_direct=True, args_schema=Request)
        def Key_Request(request:str) -> str:
            """When the guest is requesting to fix a broken item or when something is not working or needs replacement (AC, TV, coffee machine, wifi, lights, toilet, shower, etc..)."""
            return json.dumps({
                "kind":"Fix Item Request",
                "text":request
            })

    

        @tool("identify_command", return_direct=True, args_schema=Request)
        def identify_command(request):
            """Used when the guest wants something, needs something done, or is making a request or If the guest is complaining that something is lost unclean or broken"""
            command_template = '''
           Given a  guest request, your role is to classify the guest request into one of the following 7 categories:
           1- "Room Cleaning Request": When guest is requesting to clean the room.
           2- "New Item Request": When guest is requesting any extra/new items or stating that an item is missing or dirty (towel, soap, shampoo, conditioner, bodycream, pillows, tea, sugar, coffee, toothbrush, ironing board, hairdryer, slippers, etc..).
           3- "Laundry Request": When the guest requests for laundry to clean their clothes.
           4- "Late Checkout Request": When the guest requests to check out from the hotel later.
           5- "Key Request": When the guest is requesting new room keys/cards.
           6- "Fix Item Request": When the guest is requesting to fix a broken item or when something is not working or needs replacement (AC, TV, coffee machine, wifi, lights, toilet, shower, etc..).
           7- "DEFAULT": When you are unable to classify the guest request in any of the above 5 categories.

           Remember: the response can only be one of the following 7 options: ["Room Cleaning Request", "New Item Request", "Laundry Request", "Late Checkout Request", "Key Request", "Fix Item Request" , "DEFAULT"]

           <Examples>
           Request: Please clean my room
           Response: Room Cleaning Request

           Request: my pillows are dirty, i need new pillows.
           Response: New Item Request

           Request: I need my clothes washed.
           Response: Laundry Request

           Request: I forgot my keys. I need a new room card.
           Response: Key Request

           Request: The Fridge is broken. The shower is not working.
           Response: Fix Item Request

           Request: I want to store my bags
           Response: DEFAULT

           </Examples>

           Request: {input}
           Response: '''

            router_prompt = PromptTemplate(template=command_template, input_variables=["input"])
            commands_chain = LLMChain(llm=self.llm, prompt=router_prompt, verbose=False)

            response = commands_chain(request)

            return json.dumps({
                "kind": response['text'],
                "text": request
            })

        tools = [
            identify_command
          #  calc_tool
           # Room_Cleaning_Request,
           # New_Item_Request,
           # Laundry_Request,
           # Late_Checkout_Request,
           # Key_Request
        ]

        return tools

class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text=""):
        self.partialText = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.partialText += token
        if token in ["\n\n", "\n", "."]:
            print(self.partialText)
            self.partialText = ""

class Assistant():
    def __init__(self, name, tools, retriever, system_msg, chat_history=[]):
        super(Assistant, self).__init__()
        self.name = name
        self.chat_history = chat_history
        self.retriever = retriever
        self.tools = tools
        self.functions = self.init_functions()
        self.stream_handler = StreamingStdOutCallbackHandler()
        self.model = ChatOpenAI(
            model='gpt-3.5-turbo-0613',
            temperature=0.0,
            streaming=True,
            request_timeout=10,
            callbacks=[self.stream_handler]
        ).bind(functions=self.functions)
        self.system_msg = system_msg
        # self.sys_msg = (
        #  f"You are a helpful, friendly guest assistant. Your name is {self.name} and you have a cheerful very funny empathetic tone (like a close friend). Your goal is to help the guest with any questions on thier stay. "
        #  "Take your time to think before responding. Use the hotel_doc tool to see if you can give suggestions or upsell hotel offerings. "
        #  "Rephrase the guest question/statement (if needed) into a standalone question/statement. If you cannot find the answer say that 'you do not know the answer' Do not make up any information that is not in your context. "
        #  "Do not make up information that is not explicity stated. Remember: Do not makeup an answer. Respond in a cheerful, yet empathetic tone."
        # )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_msg),
                MessagesPlaceholder(variable_name='chat_history'),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.agent = (RunnablePassthrough.assign(agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"]))
            | self.prompt
            | self.model
            | OpenAIFunctionsAgentOutputParser()
        )
        self.max_iterations = 3
        self.verbose = False
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, max_iterations=self.max_iterations, verbose=self.verbose)
    
    
    def init_functions(self):
        retriever_tool = create_retriever_tool(
            self.retriever,
            "hotel_doc",
            "useful for when you want to answer questions about the hotel, it's services or to make suggestions. The input to this tool should be a complete english sentence.",
        )
        self.tools.append(retriever_tool)

        functions = [format_tool_to_openai_function(f) for f in self.tools]
        return functions

    def invoke(self, query):
        result = self.agent_executor.invoke({"input": query, "chat_history": self.chat_history})
        answer = result['output']
        self.chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ]
        )
        self.chat_history = self.chat_history[-4:]
        #print(self.chat_history)
        return answer, self.chat_history

    
def chat_with_Anna(query, hotelDoc, chat_history, system_msg):
#if __name__=="__main__":

    retriever = Document(raw_text=hotelDoc, index_name='hotel_momo').as_retriever(top_k=3)
    tools = AssistantTools().tools

    Anna = Assistant(name='Anna', tools=tools, retriever=retriever, chat_history=chat_history, system_msg=system_msg)

    return Anna.invoke(query)

