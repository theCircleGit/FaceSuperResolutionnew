import time
import requests
import concurrent.futures
import statistics
import textwrap
import argparse
from tqdm import tqdm
from celery import Celery

app = Celery('myapp', broker='amqp://guest:guest@localhost:5672')

# Configuration
HOST="192.168.1.9"
CREATE_TASK_URL = "http://192.168.1.7:3073/api/task/create"
GET_TASK_URL = "http://192.168.1.7:3073/api/task/{}"
POLL_DELAY = 0

def get_total_worker_processes():
    i = app.control.inspect()
    stats = i.stats()  # returns a dict with worker names as keys
    total_processes = 0
    if stats:
        for worker, info in stats.items():
            # The pool section typically contains 'max-concurrency'
            pool_info = info.get('pool', {})
            concurrency = pool_info.get('max-concurrency', 0)
            total_processes += concurrency
    return total_processes

def create_task():
    """
    Step 1: Create task and record the start time.
    Returns a tuple of (task_id, taskstarttime).
    """
    taskstarttime = time.time()  # Unix time when task is created
    payload = {}
    files = {
        'image': ('images.jpeg', open('/home/user/Downloads/images.jpeg', 'rb'), 'image/jpeg')
    }
    headers = {
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjNzdmOWVjZS1jOTZlLTRhN2UtYTI1NS0yMGU3OTRkZmQ3YWYiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzM5NDQ0NDI2LCJpYXQiOjE3Mzk0NDA4MjYsImVtYWlsIjoiYi5zaGFmaUBzYXQuYWUiLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7fSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJwYXNzd29yZCIsInRpbWVzdGFtcCI6MTczOTQ0MDgyNn1dLCJzZXNzaW9uX2lkIjoiNjlkNDhlNzEtODExOC00NmEzLWI2ODAtNDdjOTlmZGJhMDU2IiwiaXNfYW5vbnltb3VzIjpmYWxzZX0.zUTmiwHK-R4MAKdB9uK9QDCcB5301o_-UdfOaMmV21Q'
    }
    response = requests.request("POST", CREATE_TASK_URL, headers=headers, data = payload, files = files)
    response.raise_for_status()
    data = response.json()
    task_id = data.get("task_id")
    return task_id, taskstarttime

def poll_task(task_id):
    """
    Step 2: Poll the GET endpoint until the task is complete.
    When complete, record the end time and extract the 'proc' value.
    Returns a tuple of (taskendtime, proc).
    """
    while True:
        response = requests.get(GET_TASK_URL.format(task_id))
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "complete":
            taskendtime = time.time()  # Unix time when task completes
            # Extract the server image process time from the result
            result = data.get("result", {})
            proc = result.get("proc", 0)
            return taskendtime, proc
        time.sleep(POLL_DELAY)

def run_task():
    """
    Orchestrates the creation and polling of a task.
    Calculates:
      - totaltimefortask = taskendtime - taskstarttime
      - waittimetask = totaltimefortask - proc
    Returns a dictionary with all these values.
    """
    # Step 1: Create task and get start time.
    task_id, taskstarttime = create_task()
    # print(f"Task created: {task_id} at {taskstarttime}")

    # Step 2: Poll until the task is complete and record the end time.
    taskendtime, proc = poll_task(task_id)
    # print(f"Task completed: {task_id} at {taskendtime}")

    # Step 3: Calculate the total time taken.
    totaltimefortask = taskendtime - taskstarttime

    # Step 5: Calculate the waiting time (total time minus server processing time).
    waittimetask = totaltimefortask - proc

    return {
        "task_id": task_id,
        "taskstarttime": taskstarttime,
        "taskendtime": taskendtime,
        "totaltimefortask": totaltimefortask,
        "proc": proc,
        "waittimetask": waittimetask
    }

def main(num_tasks):
    results = []
    # Run multiple tasks concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(run_task) for _ in range(num_tasks)]
        with tqdm(total=len(futures), desc="Processing tasks") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print("Error processing task:", e)
                pbar.update(1)
    
    print("\n--- Task Results ---\n")
    box_width = 70  # total width of the box
    inner_width = box_width - 2  # width available for content (excluding borders)
    border = "+" + "-" * inner_width + "+"

    for result in results:
        # print(border)
        # # Print the task ID centered in the box
        # task_title = f"Task {result['task_id']}"
        # print("|" + task_title.center(inner_width) + "|")
        # print(border)
        # # Print each metric on its own line, left aligned
        # print("| " + f"Start Time:       {result['taskstarttime']:.3f} sec".ljust(inner_width - 1) + "|")
        # print("| " + f"End Time:         {result['taskendtime']:.3f} sec".ljust(inner_width - 1) + "|")
        # print("| " + f"Total Time:       {result['totaltimefortask']:.3f} sec".ljust(inner_width - 1) + "|")
        # print("| " + f"Server Process:   {result['proc']:.3f} sec".ljust(inner_width - 1) + "|")
        # print("| " + f"Wait Time:        {result['waittimetask']:.3f} sec".ljust(inner_width - 1) + "|")
        # print(border)
        # print()  # blank line between tasks

        # Collect data into corresponding lists for summary metrics
        total_times = [r["totaltimefortask"] for r in results]
        server_process_times = [r["proc"] for r in results]
        wait_times = [r["waittimetask"] for r in results]

        # Calculate summary statistics using the statistics module
    total_median = statistics.median(total_times)
    total_mean = statistics.mean(total_times)
    total_min = min(total_times)
    total_max = max(total_times)

    server_proc_median = statistics.median(server_process_times)
    server_proc_mean = statistics.mean(server_process_times)
    server_proc_min = min(server_process_times)
    server_proc_max = max(server_process_times)

    wait_median = statistics.median(wait_times)
    wait_mean = statistics.mean(wait_times)
    wait_min = min(wait_times)
    wait_max = max(wait_times)

    total_std = statistics.stdev(total_times) if len(total_times) > 1 else 0
    server_proc_std = statistics.stdev(server_process_times) if len(server_process_times) > 1 else 0
    wait_std = statistics.stdev(wait_times) if len(wait_times) > 1 else 0

    num_processes = get_total_worker_processes()

    box_width = 85
    border = "+" + "-" * (box_width + 2) + "+"

    def print_boxed_line(text, width=box_width):
        """Wraps text to the given width and prints each line within box borders."""
        wrapped_lines = textwrap.wrap(text, width=width)
        for line in wrapped_lines:
            print("| " + line.ljust(width) + " |")

    print(border)
    print("| " + "TEST REPORT METRICS".center(box_width) + " |")
    print(border)
    print_boxed_line(f"Total number of worker processes: {num_processes}")
    print(border)
    print_boxed_line("Total Times:")
    print_boxed_line(f"  Count: {len(total_times)}   Mean: {total_mean:.3f} sec   Median: {total_median:.3f} sec   Min: {total_min:.3f} sec   Max: {total_max:.3f} sec")
    print(border)
    print_boxed_line("Server Process Times:")
    print_boxed_line(f"  Count: {len(server_process_times)}   Mean: {server_proc_mean:.3f} sec   Median: {server_proc_median:.3f} sec   Min: {server_proc_min:.3f} sec   Max: {server_proc_max:.3f} sec")
    print(border)
    print_boxed_line("Wait Times:")
    print_boxed_line(f"  Count: {len(wait_times)}   Mean: {wait_mean:.3f} sec   Median: {wait_median:.3f} sec   Min: {wait_min:.3f} sec   Max: {wait_max:.3f} sec")
    print(border)
    print_boxed_line("Additional Metrics:")
    print_boxed_line(f"  Total Times Std Dev:       {total_std:.3f} sec")
    print_boxed_line(f"  Server Proc Times Std Dev: {server_proc_std:.3f} sec")
    print_boxed_line(f"  Wait Times Std Dev:        {wait_std:.3f} sec")
    print(border)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run tasks concurrently and generate report metrics."
    )
    parser.add_argument(
        '-n', '--num_tasks',
        type=int,
        required=True,
        help="Number of tasks to run concurrently (default: 10)"
    )
    args = parser.parse_args()
    main(args.num_tasks)
