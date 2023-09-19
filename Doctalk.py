import os
import openai
import sys

sys.path.append("../..")

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("docs/testdoc/MachineLearning.pdf")
# pages = loader.load()
# # print(len(pages))
# page = pages[0]
# page_content = page.page_content[0:500]
# print(page_content)
# print(page.metadata)

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir = "docs/youtube/"
loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
docs = loader.load()
print(docs[0].page_content[0:500])


# # for the Urls any Urls on the web we can use to load the document.

# from langchain.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

# docs = loader.load()
# print(docs[0].page_content[:500])

# # for chating with the fNotion DAtabase
# from langchain.document_loaders import NotionDirectoryLoader
# loader = NotionDirectoryLoader("docs/Notion_DB")
# docs = loader.load()

# print(docs[0].page_content[0:200])
