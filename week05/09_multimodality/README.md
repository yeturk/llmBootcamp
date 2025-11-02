## Overview
Multimodality refers to the ability to work with data that comes in different forms, such as text, audio, images, and video. Multimodality can appear in various components, allowing models and systems to handle and process a mix of these data types seamlessly.

## What kind of multimodality is supported?
### Inputs
Some models can accept multimodal inputs, such as images, audio, video, or files. The types of multimodal inputs supported depend on the model provider. For instance, OpenAI, Anthropic, and Google Gemini support documents like PDFs as inputs.
- For example, to pass an image to a chat model as URL:
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "url",
            "url": "https://...",
        },
    ],
)
response = model.invoke([message])
```
- image as in-line data
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "base64",
            "data": "<base64 string>",
            "mime_type": "image/jpeg",
        },
    ],
)
response = model.invoke([message])
```

### Outputs
Some chat models support multimodal outputs, such as images and audio. Multimodal outputs will appear as part of the AIMessage response object. See for example:

- Generating audio outputs with OpenAI;
- Generating image outputs with Google Gemini.

## Multimodality in embedding models
The current embedding interface used in LangChain is optimized entirely for text-based data, and will not work with multimodal data.

As use cases involving multimodal search and retrieval tasks become more common, we expect to expand the embedding interface to accommodate other data types like images, audio, and video.

## Multimodality in vector stores
 Similar to embeddings, vector stores are currently optimized for text-based data.

As use cases involving multimodal search and retrieval tasks become more common, we expect to expand the vector store interface to accommodate other data types like images, audio, and video.