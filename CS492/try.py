import requests
import os

from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint, set_default_backend
from sglang.srt.utils import load_image
from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

port = 30000
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))

@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))
    print(s)


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)
print(states)
for i, state in enumerate(states):
    print(states[i])
    # print_highlight(f"Answer {i+1}: {states[i]['answer']}")