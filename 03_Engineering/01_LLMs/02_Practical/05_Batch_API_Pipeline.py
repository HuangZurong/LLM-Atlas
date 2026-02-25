import json
from openai import OpenAI

"""
Industrial Best Practice: Batch API for Cost & Throughput
---------------------------------------------------------
For non-real-time tasks (classification, data extraction, summary of millions of docs),
use the Batch API to:
1. Save 50% on token costs.
2. Avoid Rate Limits (higher throughput quotas).
3. Scale horizontally without managing concurrent threads.
"""

client = OpenAI(api_key="sk-...")

def create_batch_file(tasks):
    """
    Creates a .jsonl file for the Batch API.
    Each line must be a valid Request object.
    """
    with open("batch_tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": task}],
                    "max_tokens": 1000
                }
            }
            f.write(json.dumps(request) + "\n")

def submit_batch():
    # 1. Upload file
    batch_file = client.files.create(
        file=open("batch_tasks.jsonl", "rb"),
        purpose="batch"
    )

    # 2. Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h", # Currently the only supported window
        metadata={"description": "Nightly data processing"}
    )

    print(f"Batch Job Created: {batch_job.id}")
    return batch_job.id

def check_status(batch_id):
    status = client.batches.retrieve(batch_id)
    print(f"Status: {status.status}")
    if status.status == "completed":
        print(f"Output File ID: {status.output_file_id}")
        # Use client.files.content(status.output_file_id) to download
    return status

if __name__ == "__main__":
    tasks = [
        "Classify this as SPAM or NOT: Buy cheap watches!",
        "Summarize the history of Rome in 1 sentence.",
        "What is 2+2?"
    ]
    # create_batch_file(tasks)
    # submit_batch()
    print("Batch API Pipeline structure ready.")
