import time
import openai
from utils.misc_utils import read_json_file
import logging

gpt_config_file = read_json_file("openai_utils/gpt_config.json")


# Setup OpenAI client
client = openai.OpenAI(api_key="api_key")


async def openai_request(payload, response_format, retries=2):
    """
    Predicts the sentiment of employee comments using Azure OpenAI and returns the prediction along with a success flag.

    Args:
        payload (list): The messages payload to send to the API.
        response_format (str): The expected response format.
        retries (int): Number of times to retry on failure (default: 2).

    Returns:
        tuple: (True, prediction) if successful, or (False, error message) on failure.
    """
    wait_time = 0.1  # Initial wait time for rate limit errors (exponential backoff)

    for attempt in range(1, retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=payload,
                max_tokens=gpt_config_file["max_tokens"],
                n=gpt_config_file["n"],
                seed=gpt_config_file["seed"],
                temperature=gpt_config_file["temperature"],
                response_format=response_format,
                logprobs= True,
                top_logprobs=3,
            )
            prediction = completion.choices[0].message.parsed
            return True, prediction

        except openai.RateLimitError as e:
            error_message = str(e)
            logging.error(f"Attempt {attempt}: RateLimitError: {error_message}")
            print(f"Attempt {attempt}: OpenAI API request exceeded rate limit: {error_message}")
            if attempt < retries:
                print(f"Sleeping for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                wait_time *= 2  # Double wait time for next retry
            else:
                return False, f"RateLimitError: {error_message}"

        except openai.APIError as e:
            error_message = str(e)
            # Check if the error is due to content filtering.
            if "content_filter" in error_message:
                logging.error(f"Attempt {attempt}: Content Filter Error: {error_message}")
                print(f"Attempt {attempt}: OpenAI API returned a content filter error: {error_message}")
                if attempt < retries:
                    time.sleep(1)
                else:
                    return False, f"ContentFilterError: {error_message}"
            else:
                logging.error(f"Attempt {attempt}: APIError: {error_message}")
                print(f"Attempt {attempt}: OpenAI API returned an API Error: {error_message}")
                if attempt < retries:
                    time.sleep(1)
                else:
                    return False, f"APIError: {error_message}"

        except openai.APIConnectionError as e:
            error_message = str(e)
            logging.error(f"Attempt {attempt}: APIConnectionError: {error_message}")
            print(f"Attempt {attempt}: Failed to connect to OpenAI API: {error_message}")
            if attempt < retries:
                time.sleep(1)
            else:
                return False, f"APIConnectionError: {error_message}"

        except Exception as e:
            error_message = str(e)
            logging.error(f"Attempt {attempt}: Unexpected error: {error_message}")
            print(f"Attempt {attempt}: An unexpected error occurred: {error_message}")
            if attempt < retries:
                time.sleep(1)
            else:
                return False, f"Unexpected error: {error_message}"

    return False, "All retries exhausted."

