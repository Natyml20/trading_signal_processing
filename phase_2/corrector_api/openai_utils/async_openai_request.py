import asyncio
import yaml
import aiohttp
import json
import logging
import re
import tiktoken
import time
import json
import pandas as pd
from typing import List, Dict, Any
from pandas import DataFrame

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, get_bearer_token_provider
from azure.identity import AzureCliCredential

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        payload = request_json["payload"]
        max_tokens = payload.get("max_tokens", 15)
        n = payload.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["payload"]["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["payload"]["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["payload"]["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')

async def process_api_requests(
    requests: list,
    azure_endpoint: str,
    azure_ad_token_provider: callable,
    api_version: str,
    model_name: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 2
    seconds_to_sleep_each_loop = 0.0001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call
    results = []

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    request_iterator = iter(requests)
    file_not_finished = True  # after list is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # Create the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version,
    )

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    request_json = next(request_iterator)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, "chat/completions", token_encoding_name),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if list runs out, set flag to stop reading it
                    logging.debug("Request list exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        client=client,
                        model_name=model_name,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                        results=results
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"Parallel processing complete.")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")
    
    return results

# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        client: AzureOpenAI,
        model_name: str,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        results: list,
    ):
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            survey_id = self.request_json["id"]
            payload = self.request_json["payload"]
            payload["model"] = model_name
            # Wrap the synchronous API call in asyncio.to_thread so that it runs concurrently
            completion = await asyncio.to_thread(client.chat.completions.create, **payload)
            response = completion.model_dump()
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Errors: {self.result}")
                result_data = (
                    survey_id,
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                results.append(result_data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            result_data = (
                survey_id,
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            results.append(result_data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} completed")

def parse_json_files_sentiment_gpt(json_gpt_output: List[Dict[str, Any]]) -> DataFrame:
    """
    Parses a list of JSON formatted GPT outputs, extracting specific information to create a DataFrame.

    Args:
        json_gpt_output (List[Dict[str, Any]]): A list of dictionaries, each containing GPT response data.

    Returns:
        DataFrame: A pandas DataFrame containing unique identifiers and corresponding sentiment texts.

    Raises:
        KeyError: If a required key is missing in the JSON structure.
        IndexError: If the data access exceeds the list indices.
        ValueError: If there are issues in converting the data to a DataFrame.

    Examples:
        json_gpt_output = [
            [
                None,
                [
                    {'id': '123', 'other_info': 'data'},
                    {'choices': [{'message': {'content': 'Positive sentiment detected'}}]}
                ]
            ],
            [
                None,
                [
                    {'id': '124', 'other_info': 'data'},
                    {'choices': [{'message': {'content': 'Negative sentiment detected'}}]}
                ]
            ]
        ]
        DataFrame:
            unique_idx  |  sentiment
            ----------- |  ----------------------
            123         |  Positive sentiment detected
            124         |  Negative sentiment detected
    """
    summaries = []
    try:
        for detections in json_gpt_output:
            individual_detection = {}
            # Accessing the structure based on the assumed format.
            gpt_detection = detections[1][1]
            alt_id = detections[1][0]['id']
            
            # Extract the actual sentiment content.
            gpt_detection = gpt_detection['choices'][0]['message']['content']
            individual_detection['conditional_prompt']=alt_id
            individual_detection['summary']= gpt_detection
            summaries.append(individual_detection)

        # Creating the DataFrame from the parsed data.

    except KeyError as e:
        raise KeyError(f"Missing key in JSON data structure: {e}")
    except IndexError as e:
        raise IndexError(f"Data access error: {e}")
    except ValueError as e:
        raise ValueError(f"Error creating DataFrame: {e}")

    return summaries


def parse_json_with_sumaries(json_gpt_output: List[Dict[str, Any]]) -> DataFrame:
    """
    Parses a list of JSON formatted GPT outputs, extracting specific information to create a DataFrame.

    Args:
        json_gpt_output (List[Dict[str, Any]]): A list of dictionaries, each containing GPT response data.

    Returns:
        DataFrame: A pandas DataFrame containing unique identifiers and corresponding sentiment texts.

    Raises:
        KeyError: If a required key is missing in the JSON structure.
        IndexError: If the data access exceeds the list indices.
        ValueError: If there are issues in converting the data to a DataFrame.

    Examples:
        json_gpt_output = [
            [
                None,
                [
                    {'id': '123', 'other_info': 'data'},
                    {'choices': [{'message': {'content': 'Positive sentiment detected'}}]}
                ]
            ],
            [
                None,
                [
                    {'id': '124', 'other_info': 'data'},
                    {'choices': [{'message': {'content': 'Negative sentiment detected'}}]}
                ]
            ]
        ]
        DataFrame:
            unique_idx  |  sentiment
            ----------- |  ----------------------
            123         |  Positive sentiment detected
            124         |  Negative sentiment detected
    """
    summaries = []

    try:
        for detections in json_gpt_output:
            individual_detection = {}
            # Accessing the structure based on the assumed format.
            gpt_detection = detections[1][1]
            alt_id = detections[1][0]['id']
            
            # Extract the actual sentiment content.
            gpt_detection = gpt_detection['choices'][0]['message']['content']
            individual_detection['top_percentile']=alt_id
            individual_detection['summary']= gpt_detection
            summaries.append(individual_detection)

        # Creating the DataFrame from the parsed data.

    except KeyError as e:
        raise KeyError(f"Missing key in JSON data structure: {e}")
    except IndexError as e:
        raise IndexError(f"Data access error: {e}")
    except ValueError as e:
        raise ValueError(f"Error creating DataFrame: {e}")

    return summaries



