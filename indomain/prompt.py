def HypothesisOutput(user_query, retrieval_chunks):
    format = f"""
    ### Task Description:
    You are an expert in company bylaws. Please craft a detailed response to the following query, 
    adhering to the provided answer requirements. You may reference the retrieval texts to support your response.

    ### Answer Requirements:
    1) Take your time to carefully think through each step before answering. Do not overlook any critical steps.
    2) Provide a comprehensive analysis of the issue, engaging in exploratory thinking to cover all relevant aspects.
    3) Ensure that your response addresses all aspects of the user query and refer to the retrieval texts where applicable.

    ### User Query:
    {user_query}

    ### Retrieval Texts:
    {str(retrieval_chunks)}

    ### Final Knowledge Framework:
    [Your Hypothesis Ouput here.]
    """

    return format


def ConquerAndSolve(user_query, retrieval_chunks):
    format = f"""
    ### Task Description:
    You are an expert in company bylaws. Your task is to help the user address the issue in the query 
    by breaking down the solution into clear, step-by-step actions. You may refer to the retrieval texts 
    for guidance, ensuring each step aligns with the company’s bylaws.

    ### Steps to Solve:
    1) **Understand the Context:** First, identify the key issue in the user query and the context in which it arises. What are the main points that need attention?
    2) **Review Relevant Sections of the Bylaws:** Analyze the retrieval texts to find relevant clauses or sections that pertain to the issue. Be sure to cross-reference these with the user query.
    3) **Evaluate the Legal and Procedural Requirements:** Based on the bylaws, evaluate what legal or procedural actions are required in response to the situation presented in the query.
    4) **Identify Possible Solutions:** Propose potential actions the board or shareholders could take to resolve the issue, ensuring each is in accordance with the bylaws.
    5) **Document the Decision Process:** Outline how the decision should be documented and communicated to ensure compliance and transparency.

    ### User Query:
    {user_query}

    ### Retrieval Texts:
    {str(retrieval_chunks)}

    ### Final Knowledge Framework:
    [Your Conquer And Solve Steps here.]
    """

    return format


def KnowledgeIntegrationPrompt(user_query, paths):
    format = f"""
    ### Task Description:
    You are tasked with integrating multiple paths into a comprehensive set of knowledge based on the user query. Review the paths provided, and select the most relevant and actionable information to answer the query. Ensure that only the information directly applicable to the user’s needs is retained.

    ### Steps for Knowledge Integration:
    1) **Understand the User Query:** Carefully analyze the user query to understand the core issue or question. Identify what key information is required to answer the query effectively.
    
    2) **Review All Paths:** Examine each path carefully and determine its relevance to the query. Only paths that provide useful insights or solutions should be retained.
    
    3) **Categorize the Information:** Organize the selected information from the paths into clear categories that directly address the user query. Possible categories may include: legal requirements, procedural steps, or potential solutions, depending on the nature of the query.
    
    4) **Extract Actionable Information:** From each relevant path, extract actionable insights that can be used to resolve the user’s query. Discard any information that is tangential or irrelevant.
    
    5) **Synthesize the Knowledge:** Combine the relevant insights from the selected paths into a cohesive framework or a step-by-step guide. This framework should be tailored to directly address the user’s needs as specified in the query.

    6) **Provide Clear and Concise Guidance:** Based on the synthesized knowledge, offer a well-structured response to the user query. This response should be clear, actionable, and align with the company bylaws or other relevant guidelines.

    ### User Query:
    {user_query}

    ### Paths:
    {str(paths)}

    ### Final Knowledge Framework:
    [This is where the integrated knowledge framework will be presented.]
    """

    return format


def RAGprompt(user_query, knowledge):
    format = f"""
    ### Task Description:
    You are an expert assistant. Your task is to generate a response to the following user query using the provided knowledge. Synthesize the knowledge and directly answer the query as fully and accurately as possible.

    ### Steps to Answer:
    1) **Analyze the User Query:** Carefully understand the user query to identify the core question or issue being asked.
    2) **Synthesize Knowledge:** Using the provided knowledge, craft a response that directly addresses the user's query. Do not skip any critical details.
    3) **Provide Clear and Concise Response:** Present the final answer in a structured and clear manner. Ensure the response is comprehensive and actionable.

    ### User Query:
    {user_query}

    ### Knowledge:
    {str(knowledge)}

    ### Final Answer:
    [This is where the generated answer will be presented.]
    """

    return format