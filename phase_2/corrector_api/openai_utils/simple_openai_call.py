from openai import AzureOpenAI


def simple_openai_request(azure_endpoint, azure_key, api_version, model_name, payload):
    """
    Predicts the sentiment of employee comments using Azure OpenAI and returns a DataFrame.

    Args:
        azure_endpoint (str): The Azure OpenAI endpoint URL.
        azure_key (str): The Azure OpenAI API key.
        api_version (str): The Azure OpenAI API version.
        df (pd.DataFrame): DataFrame containing employee comments.
        output_file (str): Path to the JSONL file where predictions will be saved.
        max_retries (int): Maximum number of retries for failed requests.

    Returns:
        pd.DataFrame: A DataFrame with unique_idx, Text, and sentiment predictions.
    """

    # Authenticate with Azure OpenAI

    
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=azure_key,
        api_version=api_version,
    )
    
    
    # Create a chat completion request
    completion = client.chat.completions.create(
        model=model_name,
        messages=payload,
        max_tokens=100,
        n=1,
        stop=None,
        seed=1234,
        temperature=1,
    )

    # Extract the sentiment prediction from the response
    prediction = completion.choices[0].message.content.strip()

    #print(prediction)
    return prediction

 
