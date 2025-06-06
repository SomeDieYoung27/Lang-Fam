from typing import Any,Dict,List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

def get_topic(messages : List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string  
    if len(messages) == 1:
        topic = messages[-1].content

    else:
        topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                topic += f"AI: {message.content}\n"
                  
    return topic


def resolve_urls(urls_to_resolve : List[Any],id : int) -> Dict[str,str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]


    map = {}
    for index,url in enumerate(urls):
        if url not in map:
            map[url] = f"{prefix}{id}-{index}"

    return map


def insert_citation_markers(text,citations_list):
    sorted_citations = sorted(citations_list,key = lambda c : (c['end_index'],c['start_index']),reverse=True)

    modified_text = text
    for info in sorted_citations:
        end_index = info['end_index']
        marker = ""

        for segment in info['segments']:
            marker += f"[{segment['label']}]({segment['short_url']})"

        modified_text = (
            modified_text[:end_index] + marker + modified_text[end_index:]
        )


    return modified_text

def get_citations(response,resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
     
    citations = []

    if not response or response.candidates:
        return citations
    
    candidate = response.candidates[0]

    if (
        not hasattr(candidate,'grounding_metadata')
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, 'grounding_supports')
    ) :
        return citations


    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        if not hasattr(support,"segment") or support.segment is None:
            continue


        start_index = (
            support.segment.start_index if hasattr(support.segment, 'start_index') else 0
        )

        if support.segment.end_index is None:
            continue


        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []

        if (
            hasattr(support,"grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for index in support.grounding_chunk_indices:
                try:
                     chunk = candidate.grounding_metadata.grounding_chunks[index]
                     resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                     citation["segments"].append(
                          {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                     )

                except (IndexError,AttributeError,NameError):


                    pass



        citations.append(citation)


        return citations