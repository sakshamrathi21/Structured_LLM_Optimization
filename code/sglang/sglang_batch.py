import time

from sglang import assistant, function, gen, system, user
from sglang import RuntimeEndpoint, set_default_backend
from sglang.utils import print_highlight

from system_prompt import system_prompt, queries

port = 30000
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))


@function
def text_qa(s, question):
    s+= system(system_prompt)
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": query}
        for query in queries
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
